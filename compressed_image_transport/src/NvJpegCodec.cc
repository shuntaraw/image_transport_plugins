#include <compressed_image_transport/NvJpegCodec.h>
#include <cuda_runtime_api.h>
#include <string>
#include <vector>

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

namespace compressed_image_transport {
namespace {
template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n", file, line,
            static_cast<unsigned int>(result), func);
    cudaDeviceReset();
    exit(EXIT_FAILURE);
  }
}
}  // namespace

NvJpegCodec::NvJpegCodec() {
  device_id_ = 0;
  checkCudaErrors(cudaSetDevice(device_id_));
  checkCudaErrors(nvjpegCreateSimple(&nvjpeg_handle_));
  checkCudaErrors(nvjpegJpegStateCreate(nvjpeg_handle_, &nvjpeg_state_));
  checkCudaErrors(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
}

NvJpegCodec::~NvJpegCodec() {
  ReleaseBuffers();
  checkCudaErrors(cudaStreamDestroy(stream_));
  checkCudaErrors(nvjpegJpegStateDestroy(nvjpeg_state_));
  checkCudaErrors(nvjpegDestroy(nvjpeg_handle_));
}

int NvJpegCodec::Decode(const std::vector<unsigned char> &jpeg_data,
                        nvjpegOutputFormat_t format, int *width, int *height,
                        std::vector<unsigned char> *rgb_data) {
  if (PrepareBuffers(jpeg_data, format, width, height)) {
    return EXIT_FAILURE;
  }
  checkCudaErrors(nvjpegDecode(nvjpeg_handle_, nvjpeg_state_, jpeg_data.data(),
                               jpeg_data.size(), format, &ibuf_, stream_));
  checkCudaErrors(cudaStreamSynchronize(stream_));
  int nchannels;
  switch (format) {
    case NVJPEG_OUTPUT_RGB:
    case NVJPEG_OUTPUT_BGR:
      nchannels = 3;
      break;
    case NVJPEG_OUTPUT_Y:
      nchannels = 1;
      break;
  }
  CopyToHost(*width, *height, nchannels, rgb_data);
}

void NvJpegCodec::CopyToHost(int width, int height, int nchannels,
                             std::vector<unsigned char> *rgb_data) {
  for (int c = 0; c < nchannels; ++c) {
    checkCudaErrors(cudaMemcpy2D(rgb_data_[c].data(), width, ibuf_.channel[c],
                                 width, width, height, cudaMemcpyDeviceToHost));
  }
  rgb_data->resize(width * height * nchannels);
  unsigned char *rgb = rgb_data->data();
  for (int y = 0, i = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x, ++i) {
      for (int c = 0; c < nchannels; ++c) {
        rgb[nchannels * i + c] = rgb_data_[c][i];
      }
    }
  }
}

// prepare buffers for RGBi output format
int NvJpegCodec::PrepareBuffers(const std::vector<unsigned char> &jpeg_data,
                                nvjpegOutputFormat_t format, int *width,
                                int *height) {
  int widths[NVJPEG_MAX_COMPONENT];
  int heights[NVJPEG_MAX_COMPONENT];
  int nchannels;
  nvjpegChromaSubsampling_t subsampling;
  checkCudaErrors(nvjpegGetImageInfo(
      nvjpeg_handle_, (unsigned char *)jpeg_data.data(), jpeg_data.size(),
      &nchannels, &subsampling, widths, heights));
  *width = widths[0];
  *height = heights[0];
  switch (format) {
    case NVJPEG_OUTPUT_RGB:
    case NVJPEG_OUTPUT_BGR:
      nchannels = 3;
      break;
    case NVJPEG_OUTPUT_Y:
      nchannels = 1;
      break;
  }

  // Realloc output buffer if required
  for (int c = 0; c < nchannels; c++) {
    if (ibuf_.pitch[c] == *width) {
      continue;
    }
    ibuf_.pitch[c] = *width;
    if (ibuf_.channel[c]) {
      checkCudaErrors(cudaFree(ibuf_.channel[c]));
    }
    checkCudaErrors(cudaMalloc(&ibuf_.channel[c], *width * *height));
    rgb_data_[c].resize(*width * *height);
  }
  return EXIT_SUCCESS;
}

void NvJpegCodec::ReleaseBuffers() {
  for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
    if (ibuf_.channel[c]) {
      checkCudaErrors(cudaFree(ibuf_.channel[c]));
    }
  }
}
}  // namespace compressed_image_transport
