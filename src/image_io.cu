// image_io.cu
//
// Implementation of the image I/O helpers declared in image_io.h. The stb
// single-header libraries are compiled here (and only here) by defining their
// implementation macros before inclusion.

#include "image_io.h"

#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

bool CudaCheck(cudaError_t status, const char* call, const char* file,
               int line) {
  if (status == cudaSuccess) {
    return true;
  }
  std::fprintf(stderr, "[CUDA error] %s (%d) at %s:%d -> %s\n",
               cudaGetErrorString(status), static_cast<int>(status), file, line,
               call);
  return false;
}

unsigned char* LoadImage(const std::string& path, int* width, int* height,
                         int* channels) {
  int w = 0;
  int h = 0;
  int c = 0;
  // Force three channels so the rest of the pipeline can assume RGB interleaved
  // input regardless of the source format (grayscale, RGBA, palette, ...).
  unsigned char* data = stbi_load(path.c_str(), &w, &h, &c, 3);
  if (data == nullptr) {
    std::fprintf(stderr, "[image_io] failed to load '%s': %s\n", path.c_str(),
                 stbi_failure_reason());
    return nullptr;
  }
  if (width != nullptr) {
    *width = w;
  }
  if (height != nullptr) {
    *height = h;
  }
  if (channels != nullptr) {
    // Report the forced channel count (3) because that is what the buffer
    // actually holds; the original count is c.
    *channels = 3;
  }
  return data;
}

void FreeHostImage(unsigned char* data) {
  if (data != nullptr) {
    stbi_image_free(data);
  }
}

bool SaveImage(const std::string& path, const unsigned char* data, int width,
               int height, int channels) {
  const int stride_bytes = width * channels;
  int ok = stbi_write_png(path.c_str(), width, height, channels, data,
                          stride_bytes);
  if (ok == 0) {
    std::fprintf(stderr, "[image_io] failed to write '%s'\n", path.c_str());
    return false;
  }
  return true;
}

unsigned char* AllocateDeviceImage(int width, int height, int channels,
                                   int* step_out) {
  const size_t bytes =
      static_cast<size_t>(width) * height * channels * sizeof(unsigned char);
  unsigned char* device = nullptr;
  if (!CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device), bytes))) {
    return nullptr;
  }
  if (step_out != nullptr) {
    *step_out = width * channels;
  }
  return device;
}

bool CopyToDevice(unsigned char* device, int device_step,
                  const unsigned char* host, int host_step, int width,
                  int height, int channels) {
  const size_t row_bytes = static_cast<size_t>(width) * channels;
  return CUDA_CHECK(cudaMemcpy2D(device, device_step, host, host_step, row_bytes,
                                 height, cudaMemcpyHostToDevice));
}

bool CopyToHost(unsigned char* host, int host_step, const unsigned char* device,
                int device_step, int width, int height, int channels) {
  const size_t row_bytes = static_cast<size_t>(width) * channels;
  return CUDA_CHECK(cudaMemcpy2D(host, host_step, device, device_step, row_bytes,
                                 height, cudaMemcpyDeviceToHost));
}

void FreeDeviceImage(unsigned char* device) {
  if (device != nullptr) {
    cudaFree(device);
  }
}

bool SaveDeviceStage(const unsigned char* device, int device_step, int width,
                     int height, int channels, const std::string& dir,
                     const std::string& prefix, const std::string& stage) {
  const size_t host_bytes =
      static_cast<size_t>(width) * height * channels * sizeof(unsigned char);
  unsigned char* host = static_cast<unsigned char*>(std::malloc(host_bytes));
  if (host == nullptr) {
    std::fprintf(stderr, "[image_io] malloc failed for stage '%s'\n",
                 stage.c_str());
    return false;
  }

  const int host_step = width * channels;
  bool ok = CopyToHost(host, host_step, device, device_step, width, height,
                       channels);
  if (ok) {
    const std::string out_path = dir + "/" + prefix + "_" + stage + ".png";
    ok = SaveImage(out_path, host, width, height, channels);
  }

  std::free(host);
  return ok;
}
