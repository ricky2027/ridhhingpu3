// npp_pipeline.h
//
// Declares NppPipeline, a thin C++ wrapper around a selection of NVIDIA
// Performance Primitives (NPP) image-processing routines. Each public method
// maps to a single NPP operation, accepts raw device pointers, and returns an
// NppStatus so callers can decide how to react to failures.
//
// Style: Google C++ Style Guide. PascalCase methods, snake_case variables,
// two-space indentation.

#ifndef SRC_NPP_PIPELINE_H_
#define SRC_NPP_PIPELINE_H_

#include <npp.h>
#include <nppi.h>
#include <nppi_filtering_functions.h>
#include <nppi_color_conversion.h>
#include <nppi_geometry_transforms.h>
#include <nppi_threshold_and_compare_operations.h>

#include <string>

// Returns the symbolic name for an NPP status code (e.g. "NPP_SUCCESS").
const char* NppStatusName(NppStatus status);

// Checks an NppStatus and prints a diagnostic (file/line/status name) when the
// call did not succeed. Returns true on success, false otherwise.
bool CheckNpp(NppStatus status, const char* call, const char* file, int line);

// Convenience macro: evaluates an NPP expression and reports failures. It does
// not abort; it records the status so the caller may short-circuit.
#define NPP_CHECK(call)                                        \
  CheckNpp((call), #call, __FILE__, __LINE__)

class NppPipeline {
 public:
  // Initializes the NPP stream context used by all _Ctx API calls.
  NppPipeline();
  ~NppPipeline() = default;

  // Non-copyable: the class holds only configuration, but we keep value
  // semantics explicit to avoid accidental slicing of future state.
  NppPipeline(const NppPipeline&) = delete;
  NppPipeline& operator=(const NppPipeline&) = delete;

  // When set, RunFullPipeline writes intermediate stages to disk.
  void SetVerbose(bool verbose) { verbose_ = verbose; }
  bool verbose() const { return verbose_; }

  // Base name (without extension) used to label verbose intermediate outputs.
  void SetStagePrefix(const std::string& prefix) { stage_prefix_ = prefix; }

  // Directory that verbose intermediate stages are written to.
  void SetStageDir(const std::string& dir) { stage_dir_ = dir; }

  // Resize a 3-channel 8-bit image.
  //   src            : device pointer to the source image
  //   src_width/height, src_step : source geometry (step in bytes)
  //   dst            : device pointer to the (pre-allocated) destination
  //   dst_width/height, dst_step : destination geometry (step in bytes)
  NppStatus Resize(const Npp8u* src, int src_width, int src_height,
                   int src_step, Npp8u* dst, int dst_width, int dst_height,
                   int dst_step);

  // Convert a 3-channel RGB image to single-channel grayscale.
  NppStatus RgbToGray(const Npp8u* src, int src_step, Npp8u* dst, int dst_step,
                      int width, int height);

  // Apply a box (mean) filter to a single-channel image. kernel_size is the
  // square mask edge length (e.g. 5 for a 5x5 filter).
  NppStatus BoxFilter(const Npp8u* src, int src_step, Npp8u* dst, int dst_step,
                      int width, int height, int kernel_size);

  // Canny edge detection on a single-channel image.
  NppStatus CannyEdges(const Npp8u* src, int src_step, Npp8u* dst, int dst_step,
                       int width, int height, Npp16s low_threshold,
                       Npp16s high_threshold);

  // Binarize a single-channel image: values below the threshold are set to 0,
  // values above are set to 255.
  NppStatus Threshold(const Npp8u* src, int src_step, Npp8u* dst, int dst_step,
                      int width, int height, Npp8u threshold_value);

  // Chains resize -> grayscale -> box filter -> Canny edges. The final edge map
  // is returned through dst (single channel). When verbose() is set, each
  // intermediate stage is copied to host and saved to stage_dir_.
  //
  //   src            : device pointer to the original 3-channel image
  //   src_width/height, src_step : original geometry
  //   target_width/height        : resize target
  //   dst            : device pointer for the final single-channel edge map,
  //                    sized target_width x target_height
  //   dst_step       : destination step in bytes
  //   box_kernel     : box filter edge length
  //   canny_low/high : Canny thresholds
  NppStatus RunFullPipeline(const Npp8u* src, int src_width, int src_height,
                            int src_step, int target_width, int target_height,
                            Npp8u* dst, int dst_step, int box_kernel,
                            Npp16s canny_low, Npp16s canny_high);

 private:
  bool verbose_ = false;
  std::string stage_prefix_ = "image";
  std::string stage_dir_ = "data/output";

  // Stream context required by the NPP _Ctx API (the only variants available
  // from CUDA 13 onward). Populated in the constructor via
  // nppGetStreamContext().
  NppStreamContext npp_ctx_{};
};

#endif  // SRC_NPP_PIPELINE_H_
