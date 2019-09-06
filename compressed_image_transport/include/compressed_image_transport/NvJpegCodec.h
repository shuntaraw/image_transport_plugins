#ifndef COMPRESSED_IMAGE_TRANSPORT_NV_JPEG_CODEC_H_
#define COMPRESSED_IMAGE_TRANSPORT_NV_JPEG_CODEC_H_

#include <cuda_runtime.h>
#include <nvjpeg.h>

namespace compressed_image_transport {
class NvJpegCodec {
 public:
  NvJpegCodec();
  ~NvJpegCodec();
  int Decode(const std::vector<unsigned char> &jpeg_data,
             nvjpegOutputFormat_t format, int *width, int *height,
             std::vector<unsigned char> *rgb_data);

 private:
  void CopyToHost(int width, int height, int nchannels,
                  std::vector<unsigned char> *rgb_data);
  int PrepareBuffers(const std::vector<unsigned char> &jpeg_data,
                     nvjpegOutputFormat_t format, int *width, int *height);
  void ReleaseBuffers();

  int device_id_;
  nvjpegJpegState_t nvjpeg_state_;
  nvjpegHandle_t nvjpeg_handle_;
  cudaStream_t stream_;
  nvjpegImage_t ibuf_ = {};
  std::vector<unsigned char> rgb_data_[3];
};
}  // namespace compressed_image_transport

#endif  // COMPRESSED_IMAGE_TRANSPORT_NV_JPEG_CODEC_H_
