// image_io.h
//
// Host/device image I/O helpers built on top of stb_image / stb_image_write
// and the CUDA runtime. Loading and saving happen on the host; the Allocate /
// Copy / Free helpers manage pitched-or-flat device buffers.
//
// Every CUDA call is validated with the CUDA_CHECK macro, which prints the
// offending file and line on failure.
//
// Style: Google C++ Style Guide.

#ifndef SRC_IMAGE_IO_H_
#define SRC_IMAGE_IO_H_

#include <cstdio>
#include <cstdlib>
#include <string>

#include <cuda_runtime.h>

// Validates a cudaError_t. On failure it prints a diagnostic and returns false;
// callers decide whether to abort. Use as: if (!CUDA_CHECK(cudaMalloc(...)))...
#define CUDA_CHECK(call) \
  CudaCheck((call), #call, __FILE__, __LINE__)

bool CudaCheck(cudaError_t status, const char* call, const char* file,
               int line);

// Loads a PNG/JPG/BMP/TIF image from disk. On success returns a newly
// allocated host buffer (interleaved, 8-bit per channel) and writes the image
// geometry through the out-parameters. Returns nullptr on failure.
//
// The returned pointer must be released with FreeHostImage().
unsigned char* LoadImage(const std::string& path, int* width, int* height,
                         int* channels);

// Releases a host buffer returned by LoadImage().
void FreeHostImage(unsigned char* data);

// Saves an interleaved 8-bit image to disk as PNG.
bool SaveImage(const std::string& path, const unsigned char* data, int width,
               int height, int channels);

// Allocates a flat (non-pitched) device buffer sized width*height*channels
// bytes. Returns nullptr on failure. The caller receives the row step (bytes)
// through step_out; for a flat buffer that is width*channels.
unsigned char* AllocateDeviceImage(int width, int height, int channels,
                                   int* step_out);

// Copies an interleaved host image into a device buffer. host_step and
// device_step are the per-row byte strides of each side.
bool CopyToDevice(unsigned char* device, int device_step,
                  const unsigned char* host, int host_step, int width,
                  int height, int channels);

// Copies a device image back into a host buffer.
bool CopyToHost(unsigned char* host, int host_step, const unsigned char* device,
                int device_step, int width, int height, int channels);

// Frees a device buffer allocated with AllocateDeviceImage() (or any cudaMalloc
// / nppiMalloc buffer — it just wraps cudaFree).
void FreeDeviceImage(unsigned char* device);

// Convenience used by the verbose pipeline: copies a device stage back to host
// and writes it as "<dir>/<prefix>_<stage>.png". channels is 1 or 3. Returns
// true on success. Failures are logged but non-fatal.
bool SaveDeviceStage(const unsigned char* device, int device_step, int width,
                     int height, int channels, const std::string& dir,
                     const std::string& prefix, const std::string& stage);

#endif  // SRC_IMAGE_IO_H_
