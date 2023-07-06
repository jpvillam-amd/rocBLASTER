/*
Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#pragma once

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <chrono>

/* Helpers */
#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                                    \
  if (error != hipSuccess) {                                      \
    fprintf(stderr, "hip error: '%s'(%d) at %s:%d\n",             \
            hipGetErrorString(error), error, __FILE__, __LINE__); \
    exit(EXIT_FAILURE);                                           \
  }
#endif

#ifndef CHECK_ROCBLAS_STATUS
#define CHECK_ROCBLAS_STATUS(status)                                       \
  if (status != rocblas_status_success) {                                  \
    fprintf(stderr, "rocBLAS error: ");                                    \
    fprintf(stderr, "rocBLAS error: '%s'(%d) at %s:%d\n",                  \
            rocblas_status_to_string(status), status, __FILE__, __LINE__); \
    exit(EXIT_FAILURE);                                                    \
  }
#endif

#ifndef CHECK_ROCBLAS_ERROR
#define CHECK_ROCBLAS_ERROR(error)                        \
  if (error != rocblas_status_success) {                  \
    fprintf(stderr, "rocBLAS error: ");                   \
    if (error == rocblas_status_invalid_handle)           \
      fprintf(stderr, "rocblas_status_invalid_handle");   \
    if (error == rocblas_status_not_implemented)          \
      fprintf(stderr, " rocblas_status_not_implemented"); \
    if (error == rocblas_status_invalid_pointer)          \
      fprintf(stderr, "rocblas_status_invalid_pointer");  \
    if (error == rocblas_status_invalid_size)             \
      fprintf(stderr, "rocblas_status_invalid_size");     \
    if (error == rocblas_status_memory_error)             \
      fprintf(stderr, "rocblas_status_memory_error");     \
    if (error == rocblas_status_internal_error)           \
      fprintf(stderr, "rocblas_status_internal_error");   \
    fprintf(stderr, "\n");                                \
    exit(EXIT_FAILURE);                                   \
  }
#endif

/*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and
 * return wall time */
double get_time_us_sync(hipStream_t stream) {
  hipStreamSynchronize(stream);

  auto now = std::chrono::steady_clock::now();
  // now.time_since_epoch() is the duration since epoch
  // which is converted to microseconds
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                      now.time_since_epoch())
                      .count();
  return (static_cast<double>(duration));
};