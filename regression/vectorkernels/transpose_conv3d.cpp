//===----------------------------------------------------------------------===//
//
// (c) 2025-2026 Qualcomm Technologies, Inc. All rights reserved.
//
// See https://spdx.org/licenses/BSD-3-Clause-Clear.html for license
// information. SPDX-License-Identifier: BSD-3-Clause-Clear
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <cstddef>
#include <vector>

#include <ripple-test-suite/ripple-test-suite.h>
#include <ripple.h>

namespace {

template <typename T>
int transpose_conv3d_ripple(T *__restrict__ out_ptr,
                            const T *__restrict__ in_ptr,
                            const T *__restrict__ filt_ptr,
                            const T *__restrict__ bias_ptr,
                            const T *__restrict__ scale_ptr, int d_in, int h_in,
                            int w_in, int c_in, int d_out, int h_out, int w_out,
                            int c_out, int filt_depth, int filt_height,
                            int filt_width, int stride_d, int stride_h,
                            int stride_w, int pad_d, int pad_h, int pad_w) {
  int in_s3 = 1, in_s2 = c_in, in_s1 = w_in * in_s2, in_s0 = h_in * in_s1;
  int out_s3 = 1, out_s2 = c_out, out_s1 = w_out * out_s2,
      out_s0 = h_out * out_s1;
  int filt_s4 = 1, filt_s3 = c_out, filt_s2 = c_in * filt_s3,
      filt_s1 = filt_width * filt_s2, filt_s0 = filt_height * filt_s1;

  for (int out_d = 0; out_d < d_out; out_d++) {
    int in_d_base = out_d + pad_d;
    for (int out_h = 0; out_h < h_out; out_h++) {
      int in_h_base = out_h + pad_h;
      ripple_block_t BS = ripple_set_block_shape(0, 4);
      ripple_parallel(BS, 0);
      for (int out_w = 0; out_w < w_out; out_w++) {
        int in_w_base = out_w + pad_w;
        for (int out_c = 0; out_c < c_out; out_c++) {
          float sum = bias_ptr[out_c];
          float scale_val = scale_ptr[out_c];
          for (int fd = 0; fd < filt_depth; fd++) {
            if ((in_d_base - fd) % stride_d)
              continue;
            int in_d = (in_d_base - fd) / stride_d;
            if (in_d >= d_in || in_d < 0)
              continue;
            for (int fh = 0; fh < filt_height; fh++) {
              if ((in_h_base - fh) % stride_h)
                continue;
              int in_h = (in_h_base - fh) / stride_h;
              if (in_h >= h_in || in_h < 0)
                continue;
              for (int fw = 0; fw < filt_width; fw++) {
                if ((in_w_base - fw) % stride_w)
                  continue;
                int in_w = (in_w_base - fw) / stride_w;
                if (in_w >= w_in || in_w < 0)
                  continue;
                for (int in_c = 0; in_c < c_in; in_c++) {
                  int in_idx =
                      in_d * in_s0 + in_h * in_s1 + in_w * in_s2 + in_c * in_s3;
                  int filt_idx = fd * filt_s0 + fh * filt_s1 + fw * filt_s2 +
                                 in_c * filt_s3 + out_c * filt_s4;
                  sum += (float)in_ptr[in_idx] * (float)filt_ptr[filt_idx];
                }
              }
            }
          }
          int out_idx =
              out_d * out_s0 + out_h * out_s1 + out_w * out_s2 + out_c * out_s3;
          out_ptr[out_idx] = (T)(scale_val * sum);
        }
      }
    }
  }
  return 0;
}

template <typename T>
int transpose_conv3d_5d_ripple(
    T *__restrict__ out_ptr, const T *__restrict__ in_ptr,
    const T *__restrict__ filt_ptr, const T *__restrict__ bias_ptr,
    const T *__restrict__ scale_ptr, int b_in, int d_in, int h_in, int w_in,
    int c_in, int d_out, int h_out, int w_out, int c_out, int filt_depth,
    int filt_height, int filt_width, int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w) {
  int in_s4 = 1, in_s3 = c_in, in_s2 = w_in * in_s3, in_s1 = h_in * in_s2,
      in_s0 = d_in * in_s1;
  int out_s4 = 1, out_s3 = c_out, out_s2 = w_out * out_s3,
      out_s1 = h_out * out_s2, out_s0 = d_out * out_s1;
  int filt_s4 = 1, filt_s3 = c_out, filt_s2 = c_in * filt_s3,
      filt_s1 = filt_width * filt_s2, filt_s0 = filt_height * filt_s1;

  for (int b = 0; b < b_in; b++) {
    for (int out_d = 0; out_d < d_out; out_d++) {
      int in_d_base = out_d + pad_d;
      for (int out_h = 0; out_h < h_out; out_h++) {
        int in_h_base = out_h + pad_h;
        ripple_block_t BS = ripple_set_block_shape(0, 4);
        ripple_parallel(BS, 0);
        for (int out_w = 0; out_w < w_out; out_w++) {
          int in_w_base = out_w + pad_w;
          for (int out_c = 0; out_c < c_out; out_c++) {
            float sum = bias_ptr[out_c];
            float scale_val = scale_ptr[out_c];
            for (int fd = 0; fd < filt_depth; fd++) {
              if ((in_d_base - fd) % stride_d)
                continue;
              int in_d = (in_d_base - fd) / stride_d;
              if (in_d >= d_in || in_d < 0)
                continue;
              for (int fh = 0; fh < filt_height; fh++) {
                if ((in_h_base - fh) % stride_h)
                  continue;
                int in_h = (in_h_base - fh) / stride_h;
                if (in_h >= h_in || in_h < 0)
                  continue;
                for (int fw = 0; fw < filt_width; fw++) {
                  if ((in_w_base - fw) % stride_w)
                    continue;
                  int in_w = (in_w_base - fw) / stride_w;
                  if (in_w >= w_in || in_w < 0)
                    continue;
                  for (int in_c = 0; in_c < c_in; in_c++) {
                    int in_idx = b * in_s0 + in_d * in_s1 + in_h * in_s2 +
                                 in_w * in_s3 + in_c * in_s4;
                    int filt_idx = fd * filt_s0 + fh * filt_s1 + fw * filt_s2 +
                                   in_c * filt_s3 + out_c * filt_s4;
                    sum += (float)in_ptr[in_idx] * (float)filt_ptr[filt_idx];
                  }
                }
              }
            }
            int out_idx = b * out_s0 + out_d * out_s1 + out_h * out_s2 +
                          out_w * out_s3 + out_c * out_s4;
            out_ptr[out_idx] = (T)(scale_val * sum);
          }
        }
      }
    }
  }
  return 0;
}

template <typename T>
void scalar_transposed_convolution_3d(
    T *output, const T *input, const T *filter, const T *bias, const T *scale,
    int batches, int input_depth, int input_height, int input_width,
    int input_channels, int output_depth, int output_height, int output_width,
    int output_channels, int filter_depth, int filter_height, int filter_width,
    int depth_stride, int height_stride, int width_stride, int depth_padding,
    int height_padding, int width_padding) {
  const std::size_t output_elements = static_cast<std::size_t>(batches) *
                                      output_depth * output_height *
                                      output_width * output_channels;

  for (std::size_t index = 0; index < output_elements; ++index)
    output[index] = bias[index % output_channels];

  for (int batch = 0; batch < batches; ++batch) {
    for (int input_z = 0; input_z < input_depth; ++input_z) {
      for (int input_y = 0; input_y < input_height; ++input_y) {
        for (int input_x = 0; input_x < input_width; ++input_x) {
          for (int input_channel = 0; input_channel < input_channels;
               ++input_channel) {
            const std::size_t input_index =
                ((((static_cast<std::size_t>(batch) * input_depth + input_z) *
                       input_height +
                   input_y) *
                      input_width +
                  input_x) *
                 input_channels) +
                input_channel;
            const float input_value = static_cast<float>(input[input_index]);

            for (int filter_z = 0; filter_z < filter_depth; ++filter_z) {
              const int output_z =
                  input_z * depth_stride + filter_z - depth_padding;
              if (output_z < 0 || output_z >= output_depth)
                continue;
              for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                const int output_y =
                    input_y * height_stride + filter_y - height_padding;
                if (output_y < 0 || output_y >= output_height)
                  continue;
                for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                  const int output_x =
                      input_x * width_stride + filter_x - width_padding;
                  if (output_x < 0 || output_x >= output_width)
                    continue;
                  for (int output_channel = 0; output_channel < output_channels;
                       ++output_channel) {
                    const std::size_t filter_index =
                        ((((static_cast<std::size_t>(filter_z) * filter_height +
                            filter_y) *
                               filter_width +
                           filter_x) *
                              input_channels +
                          input_channel) *
                         output_channels) +
                        output_channel;
                    const std::size_t output_index =
                        ((((static_cast<std::size_t>(batch) * output_depth +
                            output_z) *
                               output_height +
                           output_y) *
                              output_width +
                          output_x) *
                         output_channels) +
                        output_channel;
                    output[output_index] += static_cast<T>(
                        input_value * static_cast<float>(filter[filter_index]));
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  for (std::size_t index = 0; index < output_elements; ++index)
    output[index] =
        static_cast<T>(static_cast<float>(output[index]) *
                       static_cast<float>(scale[index % output_channels]));
}

using namespace ripple_test_suite;

template <typename T, unsigned SigDigits>
class TransposeConv3DTest : public Test {
  static constexpr int D_IN = 2, H_IN = 4, W_IN = 4, C_IN = 2;
  static constexpr int D_OUT = 4, H_OUT = 4, W_OUT = 4, C_OUT = 2;
  static constexpr int FD = 2, FH = 2, FW = 2;
  static constexpr int SD = 1, SH = 1, SW = 1;
  static constexpr int PD = 1, PH = 1, PW = 1;

  std::vector<T> In, Filt, Bias, Scale, Out, OutRef;

public:
  TransposeConv3DTest(TestFramework &TF)
      : Test(TF), In(D_IN * H_IN * W_IN * C_IN, 0),
        Filt(FD * FH * FW * C_IN * C_OUT, 0), Bias(C_OUT, 0), Scale(C_OUT, 0),
        Out(D_OUT * H_OUT * W_OUT * C_OUT, 0),
        OutRef(D_OUT * H_OUT * W_OUT * C_OUT, 0) {
    for (auto &v : In)
      v = static_cast<T>(4.0 * (randn() - 0.5));
    for (auto &v : Filt)
      v = static_cast<T>(2.0 * (randn() - 0.5));
    for (auto &v : Bias)
      v = static_cast<T>(randn());
    for (auto &v : Scale)
      v = static_cast<T>(0.5 + randn());
    scalar_transposed_convolution_3d(
        &OutRef[0], &In[0], &Filt[0], &Bias[0], &Scale[0], 1, D_IN, H_IN, W_IN,
        C_IN, D_OUT, H_OUT, W_OUT, C_OUT, FD, FH, FW, SD, SH, SW, PD, PH, PW);
  }

  void run(unsigned) override {
    transpose_conv3d_ripple(&Out[0], &In[0], &Filt[0], &Bias[0], &Scale[0],
                            D_IN, H_IN, W_IN, C_IN, D_OUT, H_OUT, W_OUT, C_OUT,
                            FD, FH, FW, SD, SH, SW, PD, PH, PW);
  }

  bool verify() const override {
    return isclose((size_t)(D_OUT * H_OUT * W_OUT * C_OUT), &Out[0], &OutRef[0],
                   pow(0.1, SigDigits));
  }
};

template <typename T, unsigned SigDigits>
class TransposeConv3D5DTest : public Test {
  static constexpr int B_IN = 2;
  static constexpr int D_IN = 2, H_IN = 4, W_IN = 4, C_IN = 2;
  static constexpr int D_OUT = 4, H_OUT = 4, W_OUT = 4, C_OUT = 2;
  static constexpr int FD = 2, FH = 2, FW = 2;
  static constexpr int SD = 1, SH = 1, SW = 1;
  static constexpr int PD = 1, PH = 1, PW = 1;

  std::vector<T> In, Filt, Bias, Scale, Out, OutRef;

public:
  TransposeConv3D5DTest(TestFramework &TF)
      : Test(TF), In(B_IN * D_IN * H_IN * W_IN * C_IN, 0),
        Filt(FD * FH * FW * C_IN * C_OUT, 0), Bias(C_OUT, 0), Scale(C_OUT, 0),
        Out(B_IN * D_OUT * H_OUT * W_OUT * C_OUT, 0),
        OutRef(B_IN * D_OUT * H_OUT * W_OUT * C_OUT, 0) {
    for (auto &v : In)
      v = static_cast<T>(4.0 * (randn() - 0.5));
    for (auto &v : Filt)
      v = static_cast<T>(2.0 * (randn() - 0.5));
    for (auto &v : Bias)
      v = static_cast<T>(randn());
    for (auto &v : Scale)
      v = static_cast<T>(0.5 + randn());
    scalar_transposed_convolution_3d(&OutRef[0], &In[0], &Filt[0], &Bias[0],
                                     &Scale[0], B_IN, D_IN, H_IN, W_IN, C_IN,
                                     D_OUT, H_OUT, W_OUT, C_OUT, FD, FH, FW, SD,
                                     SH, SW, PD, PH, PW);
  }

  void run(unsigned) override {
    transpose_conv3d_5d_ripple(
        &Out[0], &In[0], &Filt[0], &Bias[0], &Scale[0], B_IN, D_IN, H_IN, W_IN,
        C_IN, D_OUT, H_OUT, W_OUT, C_OUT, FD, FH, FW, SD, SH, SW, PD, PH, PW);
  }

  bool verify() const override {
    return isclose((size_t)(B_IN * D_OUT * H_OUT * W_OUT * C_OUT), &Out[0],
                   &OutRef[0], pow(0.1, SigDigits));
  }
};

// FIXME: Ripple's SESE-fication does not currently terminate for the
// back-edges created by the padded branches in these kernels. Instantiating
// either test makes compilation fail on X86, AArch64, and Hexagon. Re-enable
// them when the compiler can vectorize these non-SESE regions.
#if 0
DefineTest<TransposeConv3DTest<float, 4>>
    TTransposeConv3DFloat("TransposeConv3D<float>");
DefineTest<TransposeConv3D5DTest<float, 4>>
    TTransposeConv3D5DFloat("TransposeConv3D5D<float>");
#endif

} // namespace
