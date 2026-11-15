// npp_pipeline.cu
//
// Implementation of NppPipeline. See npp_pipeline.h for the interface and the
// contract of each method. All NPP calls are wrapped with NPP_CHECK so that a
// failing primitive prints its symbolic status name along with the originating
// file and line.

#include "npp_pipeline.h"

#include <cstdio>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "image_io.h"

const char* NppStatusName(NppStatus status) {
  switch (status) {
    case NPP_SUCCESS:
      return "NPP_SUCCESS";
    case NPP_NOT_SUPPORTED_MODE_ERROR:
      return "NPP_NOT_SUPPORTED_MODE_ERROR";
    case NPP_INVALID_HOST_POINTER_ERROR:
      return "NPP_INVALID_HOST_POINTER_ERROR";
    case NPP_INVALID_DEVICE_POINTER_ERROR:
      return "NPP_INVALID_DEVICE_POINTER_ERROR";
    case NPP_LUT_NUMBER_OF_LEVELS_ERROR:
      return "NPP_LUT_NUMBER_OF_LEVELS_ERROR";
    case NPP_TEXTURE_BIND_ERROR:
      return "NPP_TEXTURE_BIND_ERROR";
    case NPP_WRONG_INTERSECTION_ROI_ERROR:
      return "NPP_WRONG_INTERSECTION_ROI_ERROR";
    case NPP_NOT_EVEN_STEP_ERROR:
      return "NPP_NOT_EVEN_STEP_ERROR";
    case NPP_INTERPOLATION_ERROR:
      return "NPP_INTERPOLATION_ERROR";
    case NPP_RESIZE_NO_OPERATION_ERROR:
      return "NPP_RESIZE_NO_OPERATION_ERROR";
    case NPP_STEP_ERROR:
      return "NPP_STEP_ERROR";
    case NPP_NULL_POINTER_ERROR:
      return "NPP_NULL_POINTER_ERROR";
    case NPP_SIZE_ERROR:
      return "NPP_SIZE_ERROR";
    case NPP_BAD_ARGUMENT_ERROR:
      return "NPP_BAD_ARGUMENT_ERROR";
    case NPP_MEMORY_ALLOCATION_ERR:
      return "NPP_MEMORY_ALLOCATION_ERR";
    case NPP_RANGE_ERROR:
      return "NPP_RANGE_ERROR";
    default:
      return "NPP_UNKNOWN_STATUS";
  }
}

bool CheckNpp(NppStatus status, const char* call, const char* file, int line) {
  // NPP uses negative codes for errors and positive codes for warnings.
  if (status == NPP_SUCCESS) {
    return true;
  }
  if (status > 0) {
    std::fprintf(stderr, "[NPP warning] %s (%d) at %s:%d -> %s\n",
                 NppStatusName(status), static_cast<int>(status), file, line,
                 call);
    return true;
  }
  std::fprintf(stderr, "[NPP error] %s (%d) at %s:%d -> %s\n",
               NppStatusName(status), static_cast<int>(status), file, line,
               call);
  return false;
}

NppPipeline::NppPipeline() {
  // CUDA 13 removed the nppGetStreamContext() helper, so we populate the stream
  // context manually from the current device, as described in the NPP
  // documentation. npp_ctx_ is value-initialized to zero (see the header), so
  // any fields not set below default to 0 / the default stream.
  npp_ctx_.hStream = 0;  // default stream
  cudaGetDevice(&npp_ctx_.nCudaDeviceId);
  cudaDeviceGetAttribute(&npp_ctx_.nCudaDevAttrComputeCapabilityMajor,
                         cudaDevAttrComputeCapabilityMajor,
                         npp_ctx_.nCudaDeviceId);
  cudaDeviceGetAttribute(&npp_ctx_.nCudaDevAttrComputeCapabilityMinor,
                         cudaDevAttrComputeCapabilityMinor,
                         npp_ctx_.nCudaDeviceId);
  cudaStreamGetFlags(npp_ctx_.hStream, &npp_ctx_.nStreamFlags);

  cudaDeviceProp props;
  if (cudaGetDeviceProperties(&props, npp_ctx_.nCudaDeviceId) == cudaSuccess) {
    npp_ctx_.nMultiProcessorCount = props.multiProcessorCount;
    npp_ctx_.nMaxThreadsPerMultiProcessor = props.maxThreadsPerMultiProcessor;
    npp_ctx_.nMaxThreadsPerBlock = props.maxThreadsPerBlock;
    npp_ctx_.nSharedMemPerBlock = props.sharedMemPerBlock;
  }
}

NppStatus NppPipeline::Resize(const Npp8u* src, int src_width, int src_height,
                              int src_step, Npp8u* dst, int dst_width,
                              int dst_height, int dst_step) {
  NppiSize src_size = {src_width, src_height};
  NppiRect src_roi = {0, 0, src_width, src_height};
  NppiSize dst_size = {dst_width, dst_height};
  NppiRect dst_roi = {0, 0, dst_width, dst_height};

  NppStatus status = nppiResize_8u_C3R_Ctx(
      src, src_step, src_size, src_roi, dst, dst_step, dst_size, dst_roi,
      NPPI_INTER_LINEAR, npp_ctx_);
  NPP_CHECK(status);
  return status;
}

NppStatus NppPipeline::RgbToGray(const Npp8u* src, int src_step, Npp8u* dst,
                                 int dst_step, int width, int height) {
  NppiSize roi = {width, height};
  NppStatus status =
      nppiRGBToGray_8u_C3C1R_Ctx(src, src_step, dst, dst_step, roi, npp_ctx_);
  NPP_CHECK(status);
  return status;
}

NppStatus NppPipeline::BoxFilter(const Npp8u* src, int src_step, Npp8u* dst,
                                 int dst_step, int width, int height,
                                 int kernel_size) {
  NppiSize roi = {width, height};
  NppiSize mask = {kernel_size, kernel_size};
  NppiPoint anchor = {kernel_size / 2, kernel_size / 2};

  NppStatus status = nppiFilterBox_8u_C1R_Ctx(src, src_step, dst, dst_step, roi,
                                              mask, anchor, npp_ctx_);
  NPP_CHECK(status);
  return status;
}

NppStatus NppPipeline::CannyEdges(const Npp8u* src, int src_step, Npp8u* dst,
                                  int dst_step, int width, int height,
                                  Npp16s low_threshold,
                                  Npp16s high_threshold) {
  NppiSize roi_size = {width, height};
  NppiPoint src_offset = {0, 0};

  // Query and allocate the scratch buffer NPP needs for Canny.
  int buffer_size = 0;
  NppStatus status =
      nppiFilterCannyBorderGetBufferSize(roi_size, &buffer_size);
  if (!NPP_CHECK(status)) {
    return status;
  }

  Npp8u* scratch = nullptr;
  cudaError_t cuda_status =
      cudaMalloc(reinterpret_cast<void**>(&scratch), buffer_size);
  if (cuda_status != cudaSuccess) {
    std::fprintf(stderr, "[CUDA error] cudaMalloc(canny scratch) failed: %s\n",
                 cudaGetErrorString(cuda_status));
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  status = nppiFilterCannyBorder_8u_C1R_Ctx(
      src, src_step, roi_size, src_offset, dst, dst_step, roi_size,
      NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3, low_threshold, high_threshold,
      nppiNormL2, NPP_BORDER_REPLICATE, scratch, npp_ctx_);
  NPP_CHECK(status);

  cudaFree(scratch);
  return status;
}

NppStatus NppPipeline::Threshold(const Npp8u* src, int src_step, Npp8u* dst,
                                 int dst_step, int width, int height,
                                 Npp8u threshold_value) {
  NppiSize roi = {width, height};
  // Values strictly less than the threshold become 0; values greater become
  // 255 -> a clean binarization.
  NppStatus status = nppiThreshold_LTValGTVal_8u_C1R_Ctx(
      src, src_step, dst, dst_step, roi, threshold_value, 0,
      threshold_value, 255, npp_ctx_);
  NPP_CHECK(status);
  return status;
}

NppStatus NppPipeline::RunFullPipeline(const Npp8u* src, int src_width,
                                       int src_height, int src_step,
                                       int target_width, int target_height,
                                       Npp8u* dst, int dst_step, int box_kernel,
                                       Npp16s canny_low, Npp16s canny_high) {
  NppStatus status = NPP_SUCCESS;

  // Stage buffers. All three-channel work happens on resized_rgb; everything
  // after grayscale is single-channel at target dimensions.
  Npp8u* resized_rgb = nullptr;
  Npp8u* gray = nullptr;
  Npp8u* blurred = nullptr;
  int resized_step = 0;
  int gray_step = 0;
  int blur_step = 0;

  resized_rgb =
      nppiMalloc_8u_C3(target_width, target_height, &resized_step);
  gray = nppiMalloc_8u_C1(target_width, target_height, &gray_step);
  blurred = nppiMalloc_8u_C1(target_width, target_height, &blur_step);

  if (resized_rgb == nullptr || gray == nullptr || blurred == nullptr) {
    std::fprintf(stderr,
                 "[NPP error] nppiMalloc failed while allocating pipeline "
                 "stage buffers\n");
    status = NPP_MEMORY_ALLOCATION_ERR;
    goto cleanup;
  }

  // Stage 1: resize.
  status = Resize(src, src_width, src_height, src_step, resized_rgb,
                  target_width, target_height, resized_step);
  if (status != NPP_SUCCESS) {
    goto cleanup;
  }
  if (verbose_) {
    SaveDeviceStage(resized_rgb, resized_step, target_width, target_height, 3,
                    stage_dir_, stage_prefix_, "1_resize");
  }

  // Stage 2: grayscale.
  status = RgbToGray(resized_rgb, resized_step, gray, gray_step, target_width,
                     target_height);
  if (status != NPP_SUCCESS) {
    goto cleanup;
  }
  if (verbose_) {
    SaveDeviceStage(gray, gray_step, target_width, target_height, 1, stage_dir_,
                    stage_prefix_, "2_grayscale");
  }

  // Stage 3: box filter (blur).
  status = BoxFilter(gray, gray_step, blurred, blur_step, target_width,
                     target_height, box_kernel);
  if (status != NPP_SUCCESS) {
    goto cleanup;
  }
  if (verbose_) {
    SaveDeviceStage(blurred, blur_step, target_width, target_height, 1,
                    stage_dir_, stage_prefix_, "3_blur");
  }

  // Stage 4: Canny edges -> dst.
  status = CannyEdges(blurred, blur_step, dst, dst_step, target_width,
                      target_height, canny_low, canny_high);
  if (status != NPP_SUCCESS) {
    goto cleanup;
  }
  if (verbose_) {
    SaveDeviceStage(dst, dst_step, target_width, target_height, 1, stage_dir_,
                    stage_prefix_, "4_canny");
  }

cleanup:
  if (resized_rgb != nullptr) {
    nppiFree(resized_rgb);
  }
  if (gray != nullptr) {
    nppiFree(gray);
  }
  if (blurred != nullptr) {
    nppiFree(blurred);
  }
  return status;
}
