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

#define ROCBLAS_NO_DEPRECATED_WARNINGS
#define ROCBLAS_BETA_FEATURES_API
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include "gemm_common.h"
#include <iostream>
#include <pybind11/pybind11.h>
#include <string.h>
#include <vector>

struct MNK {
    std::string ta;
    std::string tb;
    rocblas_int m;
    rocblas_int n;
    rocblas_int k;
};

struct MNKB {
    std::string ta;
    std::string tb;
    rocblas_int batches;
    rocblas_int m;
    rocblas_int n;
    rocblas_int k;
};

class rocBlasFinder {
  private:
    /* Constants */
    const int deviceId = 0;
    const rocblas_int cold_calls = 5;
    const rocblas_int hot_calls = 20;

    rocblas_operation trans_a = rocblas_operation_none;
    rocblas_operation trans_b = rocblas_operation_none;

    /* TODO: types need to be made configurable */
    const rocblas_datatype input_datatype = rocblas_datatype_f16_r;
    const rocblas_datatype compute_datatype = rocblas_datatype_f32_r;
    const rocblas_datatype output_datatype = rocblas_datatype_f16_r;

    using a_t = _Float16;
    using b_t = _Float16;
    using c_t = _Float16;

    rocblas_handle handle;
    hipStream_t stream;

  public:
    rocBlasFinder() {
        // TODO: Setup dtype, device, ect.
        CHECK_HIP_ERROR(hipSetDevice(deviceId));

        CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    }
    ~rocBlasFinder() { rocblas_destroy_handle(handle); }

    // TODO: Can combine the following functions in a smart way.
    // Genetric GEMM run
    std::string run(std::string tA, std::string tB, int m, int n, int k,
                    float alpha, float beta) {
        MNK GEMM_size = {tA, tB, m, n, k};
        // GEMM parameters
        rocblas_int lda, ldb, ldc, size_a, size_b, size_c;

        std::cout << "TransA: " << GEMM_size.ta << ", TransB: " << GEMM_size.tb
                  << ", M: " << GEMM_size.m << ", N: " << GEMM_size.n
                  << ", K: " << GEMM_size.k << std::endl;
        if (GEMM_size.ta == "T") {
            trans_a = rocblas_operation_transpose;
        }

        if (GEMM_size.tb == "T") {
            trans_b = rocblas_operation_transpose;
        }
        if (GEMM_size.ta == "N") {
            trans_a = rocblas_operation_none;
        }

        if (GEMM_size.tb == "N") {
            trans_b = rocblas_operation_none;
        }

        if (trans_a == rocblas_operation_none) {
            lda = GEMM_size.m;
            size_a = GEMM_size.k * lda;
        } else {
            lda = GEMM_size.k;
            size_a = GEMM_size.m * lda;
        }
        if (trans_b == rocblas_operation_none) {
            ldb = GEMM_size.k;
            size_b = GEMM_size.n * ldb;
        } else {
            ldb = GEMM_size.n;
            size_b = GEMM_size.k * ldb;
        }
        ldc = GEMM_size.m;
        size_c = GEMM_size.n * ldc;

        // Naming: da is in GPU (device) memory. ha is in CPU (host) memory
        std::vector<a_t> ha(size_a);
        std::vector<b_t> hb(size_b);
        std::vector<c_t> hc(size_c);

        // initial data on host
        srand(1);
        for (int i = 0; i < size_a; ++i) {
            ha[i] = static_cast<_Float16>(rand() % 17);
        }
        for (int i = 0; i < size_b; ++i) {
            hb[i] = static_cast<_Float16>(rand() % 17);
        }
        for (int i = 0; i < size_c; ++i) {
            hc[i] = static_cast<_Float16>(rand() % 17);
        }

        // allocate memory on device
        float *da, *db, *dc;
        CHECK_HIP_ERROR(hipMalloc(&da, size_a * sizeof(a_t)));
        CHECK_HIP_ERROR(hipMalloc(&db, size_b * sizeof(b_t)));
        CHECK_HIP_ERROR(hipMalloc(&dc, size_c * sizeof(c_t)));

        // copy matrices from host to device
        CHECK_HIP_ERROR(hipMemcpy(da, ha.data(), sizeof(a_t) * size_a,
                                  hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(db, hb.data(), sizeof(b_t) * size_b,
                                  hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dc, hc.data(), sizeof(c_t) * size_c,
                                  hipMemcpyHostToDevice));

#define GEMM_EX_ARGS                                                           \
    handle, trans_a, trans_b, GEMM_size.m, GEMM_size.n, GEMM_size.k, &alpha,   \
        da, input_datatype, lda, db, input_datatype, ldb, &beta, dc,           \
        output_datatype, ldc, dc, output_datatype, ldc, compute_datatype
#define rocblas_gemm_exM(...) rocblas_gemm_ex(__VA_ARGS__)

        // Get number of solutions
        rocblas_int size;
        CHECK_ROCBLAS_ERROR(rocblas_gemm_ex_get_solutions(
            GEMM_EX_ARGS, rocblas_gemm_algo_solution_index,
            rocblas_gemm_flags_none, NULL, &size));
        std::cout << size << " solution(s) found" << std::endl;

        // Fill array with list of solutions
        std::vector<rocblas_int> ary(size);
        CHECK_ROCBLAS_ERROR(rocblas_gemm_ex_get_solutions(
            GEMM_EX_ARGS, rocblas_gemm_algo_solution_index,
            rocblas_gemm_flags_none, ary.data(), &size));

        // Get default timing
        // warmup
        for (rocblas_int cc = 0; cc < cold_calls; ++cc) {
            CHECK_ROCBLAS_ERROR(rocblas_gemm_exM(GEMM_EX_ARGS,
                                                 rocblas_gemm_algo_standard, 0,
                                                 rocblas_gemm_flags_none));
        }

        double time = 0;
        double ave_time = 0;
        double ave_time_default = 0;

        // timing loop
        time = get_time_us_sync(stream); // in microseconds
        for (rocblas_int hc = 0; hc < hot_calls; ++hc) {
            CHECK_ROCBLAS_ERROR(rocblas_gemm_exM(GEMM_EX_ARGS,
                                                 rocblas_gemm_algo_standard, 0,
                                                 rocblas_gemm_flags_none));
        }
        time = get_time_us_sync(stream) - time;

        ave_time_default = time / hot_calls;
        std::cout << "Default time: " << ave_time_default << " us" << std::endl;

        // Benchmark loop
        double bestTime = std::numeric_limits<double>::max();
        rocblas_int bestSol = -1;
        for (auto sol : ary) {
            // std::cout << "Testing: " << sol << " Index: " <<
            // rocblas_gemm_algo_solution_index << std::endl;
            //  warmup
            try {
                for (rocblas_int cc = 0; cc < cold_calls; ++cc) {
                    auto ret = rocblas_gemm_exM(
                        GEMM_EX_ARGS, rocblas_gemm_algo_solution_index, sol,
                        rocblas_gemm_flags_none);
                    if (ret != rocblas_status::rocblas_status_success) {
                        throw(sol);
                    }
                }

                // timing loop
                time = get_time_us_sync(stream); // in microseconds
                for (rocblas_int hc = 0; hc < hot_calls; ++hc) {
                    auto ret = rocblas_gemm_exM(
                        GEMM_EX_ARGS, rocblas_gemm_algo_solution_index, sol,
                        rocblas_gemm_flags_none);
                    if (ret != rocblas_status::rocblas_status_success) {
                        throw(sol);
                    }
                }
                time = get_time_us_sync(stream) - time;

                // track winner
                if (time < bestTime) {
                    bestSol = sol;
                    bestTime = time;
                }
            } catch (int solc) {
                std::cout << "Error on solution: " << solc << std::endl;
            }
        }
        ave_time = bestTime / hot_calls;
        std::cout << "Winner: " << ave_time << " us "
                  << "(sol " << bestSol << ")" << std::endl;
        return "Default: " + std::to_string(ave_time_default) +
               " Winner: " + std::to_string(ave_time) +
               " Solution: " + std::to_string(bestSol);
    }

    // Strided Batched GEMM run
    // XXX: Assume C and D are the same matrix, need a new implementation if
    // we find that to be problematic.
    std::string run(std::string tA, std::string tB, int m, int n, int k,
                    float alpha, float beta, int stride_a, int stride_b,
                    int stride_c, int batches) {
        MNKB GEMM_size = {tA, tB, batches, m, n, k};
        // GEMM parameters
        rocblas_int lda, ldb, ldc, size_a, size_b, size_c;

        std::cout << "TransA: " << GEMM_size.ta << ", TransB: " << GEMM_size.tb
                  << ", M: " << GEMM_size.m << ", N: " << GEMM_size.n
                  << ", K: " << GEMM_size.k << std::endl;
        if (GEMM_size.ta == "T") {
            trans_a = rocblas_operation_transpose;
        }

        if (GEMM_size.tb == "T") {
            trans_b = rocblas_operation_transpose;
        }
        if (GEMM_size.ta == "N") {
            trans_a = rocblas_operation_none;
        }

        if (GEMM_size.tb == "N") {
            trans_b = rocblas_operation_none;
        }

        if (trans_a == rocblas_operation_none) {
            lda = GEMM_size.m;
            size_a = GEMM_size.k * lda * GEMM_size.batches;
        } else {
            lda = GEMM_size.k;
            size_a = GEMM_size.m * lda * GEMM_size.batches;
        }
        if (trans_b == rocblas_operation_none) {
            ldb = GEMM_size.k;
            size_b = GEMM_size.n * ldb * GEMM_size.batches;
        } else {
            ldb = GEMM_size.n;
            size_b = GEMM_size.k * ldb * GEMM_size.batches;
        }
        ldc = GEMM_size.m;
        size_c = GEMM_size.n * ldc;

        // Naming: da is in GPU (device) memory. ha is in CPU (host) memory
        std::vector<a_t> ha(size_a);
        std::vector<b_t> hb(size_b);
        std::vector<c_t> hc(size_c);

        // initial data on host
        srand(1);
        for (int i = 0; i < size_a; ++i) {
            ha[i] = static_cast<_Float16>(rand() % 17);
        }
        for (int i = 0; i < size_b; ++i) {
            hb[i] = static_cast<_Float16>(rand() % 17);
        }
        for (int i = 0; i < size_c; ++i) {
            hc[i] = static_cast<_Float16>(rand() % 17);
        }

        // allocate memory on device
        float *da, *db, *dc;
        CHECK_HIP_ERROR(hipMalloc(&da, size_a * sizeof(a_t)));
        CHECK_HIP_ERROR(hipMalloc(&db, size_b * sizeof(b_t)));
        CHECK_HIP_ERROR(hipMalloc(&dc, size_c * sizeof(c_t)));

        // copy matrices from host to device
        CHECK_HIP_ERROR(hipMemcpy(da, ha.data(), sizeof(a_t) * size_a,
                                  hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(db, hb.data(), sizeof(b_t) * size_b,
                                  hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dc, hc.data(), sizeof(c_t) * size_c,
                                  hipMemcpyHostToDevice));

#define GEMM_EX_ARGS                                                           \
    handle, trans_a, trans_b, GEMM_size.m, GEMM_size.n, GEMM_size.k, &alpha,   \
        da, input_datatype, lda, static_cast<rocblas_int>(stride_a), db,       \
        input_datatype, ldb, static_cast<rocblas_int>(stride_b), &beta, dc,    \
        output_datatype, ldc, static_cast<rocblas_int>(stride_c), dc,          \
        output_datatype, ldc, static_cast<rocblas_int>(stride_c),              \
        GEMM_size.batches, compute_datatype
#define rocblas_gemm_exM(...) rocblas_gemm_strided_batched_ex(__VA_ARGS__)

        // Get number of solutions
        rocblas_int size;
        CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched_ex_get_solutions(
            GEMM_EX_ARGS, rocblas_gemm_algo_solution_index,
            rocblas_gemm_flags_none, NULL, &size));
        std::cout << size << " solution(s) found" << std::endl;

        // Fill array with list of solutions
        std::vector<rocblas_int> ary(size);
        CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched_ex_get_solutions(
            GEMM_EX_ARGS, rocblas_gemm_algo_solution_index,
            rocblas_gemm_flags_none, ary.data(), &size));

        // Get default timing
        // warmup
        for (rocblas_int cc = 0; cc < cold_calls; ++cc) {
            CHECK_ROCBLAS_ERROR(rocblas_gemm_exM(GEMM_EX_ARGS,
                                                 rocblas_gemm_algo_standard, 0,
                                                 rocblas_gemm_flags_none));
        }

        double time = 0;
        double ave_time = 0;
        double ave_time_default = 0;

        // timing loop
        time = get_time_us_sync(stream); // in microseconds
        for (rocblas_int hc = 0; hc < hot_calls; ++hc) {
            CHECK_ROCBLAS_ERROR(rocblas_gemm_exM(GEMM_EX_ARGS,
                                                 rocblas_gemm_algo_standard, 0,
                                                 rocblas_gemm_flags_none));
        }
        time = get_time_us_sync(stream) - time;

        ave_time_default = time / hot_calls;
        std::cout << "Default time: " << ave_time_default << " us" << std::endl;

        // Benchmark loop
        double bestTime = std::numeric_limits<double>::max();
        rocblas_int bestSol = -1;
        for (auto sol : ary) {
            // std::cout << "Testing: " << sol << " Index: " <<
            // rocblas_gemm_algo_solution_index << std::endl;
            // warmup
            try {
                for (rocblas_int cc = 0; cc < cold_calls; ++cc) {
                    auto ret = rocblas_gemm_exM(
                        GEMM_EX_ARGS, rocblas_gemm_algo_solution_index, sol,
                        rocblas_gemm_flags_none);
                    if (ret != rocblas_status::rocblas_status_success) {
                        throw(sol);
                    }
                }

                // timing loop
                time = get_time_us_sync(stream); // in microseconds
                for (rocblas_int hc = 0; hc < hot_calls; ++hc) {
                    auto ret = rocblas_gemm_exM(
                        GEMM_EX_ARGS, rocblas_gemm_algo_solution_index, sol,
                        rocblas_gemm_flags_none);
                    if (ret != rocblas_status::rocblas_status_success) {
                        throw(sol);
                    }
                }
                time = get_time_us_sync(stream) - time;

                // track winner
                if (time < bestTime) {
                    bestSol = sol;
                    bestTime = time;
                }
            } catch (int solc) {
                std::cout << "Error on solution: " << solc << std::endl;
            }
        }
        ave_time = bestTime / hot_calls;
        std::cout << "Winner: " << ave_time << " us "
                  << "(sol " << bestSol << ")" << std::endl;
        return "Default: " + std::to_string(ave_time_default) +
               " Winner: " + std::to_string(ave_time) +
               " Solution: " + std::to_string(bestSol);
    }
};
namespace py = pybind11;
PYBIND11_MODULE(rocBlasFinder, m) {
    // bindings to rocBlasFinder class
    py::class_<rocBlasFinder>(m, "rocBlasFinder")
        .def(py::init())
        .def("run", py
             : overload_cast<std::string, std::string, int, int, int, float,
                             float>(&rocBlasFinder::run),
               "Generic GEMM")
        .def("run", py
             : overload_cast<std::string, std::string, int, int, int, float,
                             float, int, int, int, int>(&rocBlasFinder::run),
               "Strided batched GEMM");
}
/*
   int main(){
   std::vector<MNK> GEMM_sizes{
   {"N","N",2048,8192,2048},
   {"N","N",2048,   8192,    50368},
   {"N","N",2048,   8192,    6144},
   {"N","N",2048,   8192,    8192},
   {"N","N",8192,   8192,    2048},
   {"N","T",2048,   2048,    8192},
   {"N","T",2048,   50368,   8192},
   {"N","T",2048,   6144,    8192},
   {"N","T",2048,   8192,    8192},
   {"N","T",8192,   2048,    8192},
   {"T","N",2048,   8192,    2048},
   {"T","N",2048,   8192,    8192},
   {"T","N",50368,  8192,    2048},
   {"T","N",6144,   8192,    2048},
   {"T","N",8192,   8192,    2048},
   };

   rocBlasFinder bf;
   bf.run(GEMM_sizes[1]);
   }
   */
