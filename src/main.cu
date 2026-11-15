// main.cu
//
// Batch image processor. Scans an input directory for images, runs a selected
// NPP operation (or the full pipeline) on each one, records per-image GPU
// timings with CUDA events, writes a results CSV, and prints a summary.
//
// Usage: see PrintUsage() or `bin/npp_processor -h`.
//
// Style: Google C++ Style Guide.

#include <dirent.h>
#include <getopt.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <npp.h>

#include "image_io.h"
#include "npp_pipeline.h"

namespace {

// Holds parsed command-line options.
struct Options {
  std::string input_dir;
  std::string output_dir = "data/output";
  std::string pipeline = "full";
  int target_width = 512;
  int threshold_value = 128;
  bool verbose = false;
};

// One row of the results CSV.
struct ResultRow {
  std::string image_name;
  int original_width = 0;
  int original_height = 0;
  std::string operation;
  float gpu_time_ms = 0.0f;
  std::string timestamp;
};

void PrintUsage(const char* program) {
  std::printf(
      "Usage: %s -i <input_dir> [options]\n"
      "\n"
      "Batch-process images with CUDA NPP.\n"
      "\n"
      "Options:\n"
      "  -i <path>      Input directory containing images (required)\n"
      "  -o <path>      Output directory (default: data/output/)\n"
      "  -p <pipeline>  Operation: resize | grayscale | blur | canny |\n"
      "                 threshold | full (default: full)\n"
      "  -w <width>     Target resize width; height keeps aspect ratio\n"
      "                 (default: 512)\n"
      "  -t <value>     Threshold value for threshold/Canny (default: 128)\n"
      "  -v             Verbose: save intermediate pipeline stages\n"
      "  -h             Show this help and exit\n",
      program);
}

// Lowercases a copy of the input string.
std::string ToLower(const std::string& value) {
  std::string out = value;
  std::transform(out.begin(), out.end(), out.begin(),
                 [](unsigned char ch) { return std::tolower(ch); });
  return out;
}

// Returns true when the path ends with a supported image extension.
bool HasImageExtension(const std::string& name) {
  const std::string lower = ToLower(name);
  const char* extensions[] = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"};
  for (const char* ext : extensions) {
    const size_t ext_len = std::strlen(ext);
    if (lower.size() >= ext_len &&
        lower.compare(lower.size() - ext_len, ext_len, ext) == 0) {
      return true;
    }
  }
  return false;
}

// Strips any directory prefix and the final extension from a path.
std::string BaseNameNoExt(const std::string& path) {
  size_t slash = path.find_last_of("/\\");
  std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);
  size_t dot = base.find_last_of('.');
  if (dot != std::string::npos) {
    base = base.substr(0, dot);
  }
  return base;
}

// Scans a directory and returns the full paths of all image files found.
std::vector<std::string> ScanInputDirectory(const std::string& dir) {
  std::vector<std::string> paths;
  DIR* handle = opendir(dir.c_str());
  if (handle == nullptr) {
    std::fprintf(stderr, "[main] cannot open input directory '%s'\n",
                 dir.c_str());
    return paths;
  }
  struct dirent* entry = nullptr;
  while ((entry = readdir(handle)) != nullptr) {
    const std::string name = entry->d_name;
    if (name == "." || name == "..") {
      continue;
    }
    if (!HasImageExtension(name)) {
      continue;
    }
    std::string full = dir;
    if (!full.empty() && full.back() != '/') {
      full += '/';
    }
    full += name;
    paths.push_back(full);
  }
  closedir(handle);
  std::sort(paths.begin(), paths.end());
  return paths;
}

// Returns a human-readable UTC-ish local timestamp (YYYY-MM-DD HH:MM:SS).
std::string CurrentTimestamp() {
  std::time_t now = std::time(nullptr);
  std::tm tm_buf;
#if defined(_WIN32)
  localtime_s(&tm_buf, &now);
#else
  localtime_r(&now, &tm_buf);
#endif
  char buffer[32];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm_buf);
  return std::string(buffer);
}

// Prints GPU device information and the NPP library version.
void PrintDeviceInfo() {
  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "[main] cudaGetDevice failed: %s\n",
                 cudaGetErrorString(status));
    return;
  }
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
    std::fprintf(stderr, "[main] cudaGetDeviceProperties failed\n");
    return;
  }
  const double total_gb =
      static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);
  const NppLibraryVersion* npp_version = nppGetLibVersion();

  std::printf("==== GPU device info ====\n");
  std::printf("  Device        : %s\n", prop.name);
  std::printf("  Compute cap.  : %d.%d\n", prop.major, prop.minor);
  std::printf("  Total memory  : %.2f GB\n", total_gb);
  std::printf("  Multiprocs    : %d\n", prop.multiProcessorCount);
  if (npp_version != nullptr) {
    std::printf("  NPP version   : %d.%d.%d\n", npp_version->major,
                npp_version->minor, npp_version->build);
  }
  std::printf("=========================\n\n");
}

// Computes a resize target height that preserves the source aspect ratio.
int AspectHeight(int src_width, int src_height, int target_width) {
  if (src_width <= 0) {
    return target_width;
  }
  int height = static_cast<int>(static_cast<long long>(target_width) *
                                src_height / src_width);
  return std::max(1, height);
}

// Writes the results to a CSV file at the given path.
bool WriteResultsCsv(const std::string& path,
                     const std::vector<ResultRow>& rows) {
  std::FILE* file = std::fopen(path.c_str(), "w");
  if (file == nullptr) {
    std::fprintf(stderr, "[main] cannot open results file '%s'\n", path.c_str());
    return false;
  }
  std::fprintf(file,
               "image_name,original_width,original_height,operation,"
               "gpu_time_ms,timestamp\n");
  for (const ResultRow& row : rows) {
    std::fprintf(file, "%s,%d,%d,%s,%.3f,%s\n", row.image_name.c_str(),
                 row.original_width, row.original_height, row.operation.c_str(),
                 row.gpu_time_ms, row.timestamp.c_str());
  }
  std::fclose(file);
  return true;
}

// Runs the selected single operation (or the full pipeline) on one already
// uploaded device image. On success, result_device/result_channels/
// result_width/result_height describe the output that should be copied back
// and saved. Returns the NPP status of the final operation.
//
// src_device is the original uploaded RGB image (src_width x src_height).
NppStatus RunOperation(NppPipeline* pipeline, const Options& options,
                       const std::string& op, unsigned char* src_device,
                       int src_width, int src_height, int src_step,
                       unsigned char** result_device, int* result_step,
                       int* result_width, int* result_height,
                       int* result_channels) {
  const int target_width = options.target_width;
  const int target_height =
      AspectHeight(src_width, src_height, target_width);

  // Default: result is the full-size original geometry; overridden per op.
  NppStatus status = NPP_SUCCESS;

  if (op == "resize") {
    int step = 0;
    unsigned char* dst = reinterpret_cast<unsigned char*>(
        nppiMalloc_8u_C3(target_width, target_height, &step));
    status = pipeline->Resize(src_device, src_width, src_height, src_step, dst,
                              target_width, target_height, step);
    *result_device = dst;
    *result_step = step;
    *result_width = target_width;
    *result_height = target_height;
    *result_channels = 3;
    return status;
  }

  if (op == "grayscale") {
    int step = 0;
    unsigned char* dst = reinterpret_cast<unsigned char*>(
        nppiMalloc_8u_C1(src_width, src_height, &step));
    status = pipeline->RgbToGray(src_device, src_step, dst, step, src_width,
                                 src_height);
    *result_device = dst;
    *result_step = step;
    *result_width = src_width;
    *result_height = src_height;
    *result_channels = 1;
    return status;
  }

  if (op == "blur") {
    // Grayscale first, then box filter.
    int gray_step = 0;
    unsigned char* gray = reinterpret_cast<unsigned char*>(
        nppiMalloc_8u_C1(src_width, src_height, &gray_step));
    status = pipeline->RgbToGray(src_device, src_step, gray, gray_step,
                                 src_width, src_height);
    int blur_step = 0;
    unsigned char* blurred = reinterpret_cast<unsigned char*>(
        nppiMalloc_8u_C1(src_width, src_height, &blur_step));
    if (status == NPP_SUCCESS) {
      status = pipeline->BoxFilter(gray, gray_step, blurred, blur_step,
                                   src_width, src_height, 5);
    }
    nppiFree(gray);
    *result_device = blurred;
    *result_step = blur_step;
    *result_width = src_width;
    *result_height = src_height;
    *result_channels = 1;
    return status;
  }

  if (op == "canny") {
    int gray_step = 0;
    unsigned char* gray = reinterpret_cast<unsigned char*>(
        nppiMalloc_8u_C1(src_width, src_height, &gray_step));
    status = pipeline->RgbToGray(src_device, src_step, gray, gray_step,
                                 src_width, src_height);
    int edge_step = 0;
    unsigned char* edges = reinterpret_cast<unsigned char*>(
        nppiMalloc_8u_C1(src_width, src_height, &edge_step));
    if (status == NPP_SUCCESS) {
      const Npp16s low = static_cast<Npp16s>(options.threshold_value / 2);
      const Npp16s high = static_cast<Npp16s>(options.threshold_value);
      status = pipeline->CannyEdges(gray, gray_step, edges, edge_step,
                                    src_width, src_height, low, high);
    }
    nppiFree(gray);
    *result_device = edges;
    *result_step = edge_step;
    *result_width = src_width;
    *result_height = src_height;
    *result_channels = 1;
    return status;
  }

  if (op == "threshold") {
    int gray_step = 0;
    unsigned char* gray = reinterpret_cast<unsigned char*>(
        nppiMalloc_8u_C1(src_width, src_height, &gray_step));
    status = pipeline->RgbToGray(src_device, src_step, gray, gray_step,
                                 src_width, src_height);
    int bin_step = 0;
    unsigned char* binary = reinterpret_cast<unsigned char*>(
        nppiMalloc_8u_C1(src_width, src_height, &bin_step));
    if (status == NPP_SUCCESS) {
      status = pipeline->Threshold(
          gray, gray_step, binary, bin_step, src_width, src_height,
          static_cast<Npp8u>(options.threshold_value));
    }
    nppiFree(gray);
    *result_device = binary;
    *result_step = bin_step;
    *result_width = src_width;
    *result_height = src_height;
    *result_channels = 1;
    return status;
  }

  // Default: "full" -> resize -> grayscale -> blur -> canny.
  int edge_step = 0;
  unsigned char* edges = reinterpret_cast<unsigned char*>(
      nppiMalloc_8u_C1(target_width, target_height, &edge_step));
  const Npp16s low = static_cast<Npp16s>(options.threshold_value / 2);
  const Npp16s high = static_cast<Npp16s>(options.threshold_value);
  status = pipeline->RunFullPipeline(src_device, src_width, src_height, src_step,
                                     target_width, target_height, edges,
                                     edge_step, 5, low, high);
  *result_device = edges;
  *result_step = edge_step;
  *result_width = target_width;
  *result_height = target_height;
  *result_channels = 1;
  return status;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;

  int opt = 0;
  while ((opt = getopt(argc, argv, "i:o:p:w:t:vh")) != -1) {
    switch (opt) {
      case 'i':
        options.input_dir = optarg;
        break;
      case 'o':
        options.output_dir = optarg;
        break;
      case 'p':
        options.pipeline = ToLower(optarg);
        break;
      case 'w':
        options.target_width = std::atoi(optarg);
        break;
      case 't':
        options.threshold_value = std::atoi(optarg);
        break;
      case 'v':
        options.verbose = true;
        break;
      case 'h':
        PrintUsage(argv[0]);
        return 0;
      default:
        PrintUsage(argv[0]);
        return 1;
    }
  }

  if (options.input_dir.empty()) {
    std::fprintf(stderr, "[main] missing required -i <input_dir>\n\n");
    PrintUsage(argv[0]);
    return 1;
  }
  if (options.target_width <= 0) {
    std::fprintf(stderr, "[main] -w width must be positive\n");
    return 1;
  }

  PrintDeviceInfo();

  const std::vector<std::string> images = ScanInputDirectory(options.input_dir);
  if (images.empty()) {
    std::fprintf(stderr, "[main] no images found in '%s'\n",
                 options.input_dir.c_str());
    return 1;
  }
  std::printf("Found %zu image(s) in %s\n", images.size(),
              options.input_dir.c_str());
  std::printf("Operation: %s | output: %s\n\n", options.pipeline.c_str(),
              options.output_dir.c_str());

  NppPipeline pipeline;
  pipeline.SetVerbose(options.verbose);
  pipeline.SetStageDir(options.output_dir);

  std::vector<ResultRow> results;
  results.reserve(images.size());

  cudaEvent_t start_event;
  cudaEvent_t stop_event;
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);

  int processed = 0;
  int skipped = 0;
  double total_gpu_ms = 0.0;
  const int total = static_cast<int>(images.size());

  for (int index = 0; index < total; ++index) {
    const std::string& path = images[index];
    const std::string base = BaseNameNoExt(path);

    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* host = LoadImage(path, &width, &height, &channels);
    if (host == nullptr) {
      std::fprintf(stderr, "[%d/%d] WARNING: skipping unreadable image %s\n",
                   index + 1, total, path.c_str());
      ++skipped;
      continue;
    }

    // Upload original RGB image to the device.
    int src_step = 0;
    unsigned char* src_device =
        AllocateDeviceImage(width, height, channels, &src_step);
    if (src_device == nullptr) {
      std::fprintf(stderr, "[%d/%d] WARNING: device alloc failed for %s\n",
                   index + 1, total, path.c_str());
      FreeHostImage(host);
      ++skipped;
      continue;
    }
    CopyToDevice(src_device, src_step, host, width * channels, width, height,
                 channels);

    pipeline.SetStagePrefix(base);

    // Time the operation with CUDA events.
    unsigned char* result_device = nullptr;
    int result_step = 0;
    int result_width = 0;
    int result_height = 0;
    int result_channels = 0;

    cudaEventRecord(start_event);
    NppStatus status = RunOperation(
        &pipeline, options, options.pipeline, src_device, width, height,
        src_step, &result_device, &result_step, &result_width, &result_height,
        &result_channels);
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, start_event, stop_event);

    if (status != NPP_SUCCESS || result_device == nullptr) {
      std::fprintf(stderr, "[%d/%d] WARNING: processing failed for %s\n",
                   index + 1, total, path.c_str());
      FreeDeviceImage(src_device);
      if (result_device != nullptr) {
        nppiFree(result_device);
      }
      FreeHostImage(host);
      ++skipped;
      continue;
    }

    // Copy result back to host and save.
    const size_t out_bytes =
        static_cast<size_t>(result_width) * result_height * result_channels;
    unsigned char* out_host =
        static_cast<unsigned char*>(std::malloc(out_bytes));
    bool saved = false;
    if (out_host != nullptr) {
      CopyToHost(out_host, result_width * result_channels, result_device,
                 result_step, result_width, result_height, result_channels);
      const std::string out_path = options.output_dir + "/" + base + "_" +
                                   options.pipeline + ".png";
      saved = SaveImage(out_path, out_host, result_width, result_height,
                        result_channels);
      std::free(out_host);
    }

    // Record result and progress.
    ResultRow row;
    row.image_name = base;
    row.original_width = width;
    row.original_height = height;
    row.operation = options.pipeline;
    row.gpu_time_ms = gpu_ms;
    row.timestamp = CurrentTimestamp();
    results.push_back(row);

    total_gpu_ms += gpu_ms;
    ++processed;
    std::printf("[%d/%d] processed %s%s — %.2fms\n", index + 1, total,
                base.c_str(), saved ? "" : " (save failed)", gpu_ms);

    // Clean up per-image device memory.
    nppiFree(result_device);
    FreeDeviceImage(src_device);
    FreeHostImage(host);
  }

  cudaEventDestroy(start_event);
  cudaEventDestroy(stop_event);

  // Persist results.
  const std::string csv_path = "data/results/execution_log.csv";
  if (WriteResultsCsv(csv_path, results)) {
    std::printf("\nResults written to %s\n", csv_path.c_str());
  }

  // Summary.
  const double avg_ms =
      processed > 0 ? total_gpu_ms / processed : 0.0;
  const double throughput =
      total_gpu_ms > 0.0 ? (processed * 1000.0 / total_gpu_ms) : 0.0;

  std::printf("\n==== Summary ====\n");
  std::printf("  Images processed : %d\n", processed);
  std::printf("  Images skipped   : %d\n", skipped);
  std::printf("  Total GPU time   : %.2f ms\n", total_gpu_ms);
  std::printf("  Average per image: %.2f ms\n", avg_ms);
  std::printf("  Throughput       : %.2f images/sec (GPU time)\n", throughput);
  std::printf("=================\n");

  return 0;
}
