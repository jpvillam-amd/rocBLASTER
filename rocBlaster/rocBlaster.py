#!/usr/bin/env python
"""
Simple programs that gets all BLAS GEMMs from an executable and tries to tune them.
"""
import argparse
import subprocess
import os
import re
import csv

# TODO: Need to figure out this relative path
from rocBlasFinder import rocBlasFinder


class GEMM:
    """
    Class to contain a gemm and its occurances.
    """

    STRIDED_BATCHED_ROCBLAS_BENCH_RE = (
        r"./rocblas-bench -f gemm_strided_batched_ex"
        r" --transposeA (?P<TRANSPOSE_A>\w)"
        r" --transposeB (?P<TRANSPOSE_B>\w)"
        r" -m (?P<M>\d+)"
        r" -n (?P<N>\d+)"
        r" -k (?P<K>\d+)"
        r" --alpha (?P<ALPHA>\d+)"
        r" --a_type (?P<A_TYPE>\w+)"
        r" --lda (?P<LDA>\d+)"
        r" --stride_a (?P<STRIDE_A>\d+)"
        r" --b_type (?P<B_TYPE>\w+)"
        r" --ldb (?P<LDB>\d+)"
        r" --stride_b (?P<STRIDE_B>\d+)"
        r" --beta (?P<BETA>\d+)"
        r" --c_type (?P<C_TYPE>\w+)"
        r" --ldc (?P<LDC>\d+)"
        r" --stride_c (?P<STRIDE_C>\d+)"
        r" --d_type (?P<D_TYPE>\w+)"
        r" --ldd (?P<LDD>\d+)"
        r" --stride_d (?P<STRIDE_D>\d+)"
        r" --batch_count (?P<BATCH_COUNT>\d+)"
        r" --compute_type (?P<COMPUTE_TYPE>\w+)"
        r" --algo (?P<ALGO>\d+)"
        r" --solution_index (?P<SOLUTION_INDEX>\d+)"
        r" --flags (?P<FLAGS>\w+)"
    )

    GENERIC_ROCBLAS_BENCH_RE = (
        r"./rocblas-bench -f gemm_ex"
        r" --transposeA (?P<TRANSPOSE_A>\w)"
        r" --transposeB (?P<TRANSPOSE_B>\w)"
        r" -m (?P<M>\d+)"
        r" -n (?P<N>\d+)"
        r" -k (?P<K>\d+)"
        r" --alpha (?P<ALPHA>\d+)"
        r" --a_type (?P<A_TYPE>\w+)"
        r" --lda (?P<LDA>\d+)"
        r" --b_type (?P<B_TYPE>\w+)"
        r" --ldb (?P<LDB>\d+)"
        r" --beta (?P<BETA>\d+)"
        r" --c_type (?P<C_TYPE>\w+)"
        r" --ldc (?P<LDC>\d+)"
        r" --d_type (?P<D_TYPE>\w+)"
        r" --ldd (?P<LDD>\d+)"
        r" --compute_type (?P<COMPUTE_TYPE>\w+)"
        r" --algo (?P<ALGO>\d+)"
        r" --solution_index (?P<SOLUTION_INDEX>\d+)"
        r" --flags (?P<FLAGS>\w+)"
    )

    def __init__(self, rocblas_bench_string):
        if match := re.match(self.GENERIC_ROCBLAS_BENCH_RE, rocblas_bench_string):
            self.match = match
            self.gemm_type = "Generic"
            self.count = 1
            self.tA = self.match.group("TRANSPOSE_A")
            self.tB = self.match.group("TRANSPOSE_B")
            self.m = int(self.match.group("M"))
            self.n = int(self.match.group("N"))
            self.k = int(self.match.group("K"))
            self.alpha = float(self.match.group("ALPHA"))
            self.lda = int(self.match.group("LDA"))
            self.ldb = int(self.match.group("LDB"))
            self.beta = float(self.match.group("BETA"))
            self.ldc = int(self.match.group("LDC"))
            self.compute_type = self.match.group("COMPUTE_TYPE")
            self.a_type = self.match.group("A_TYPE")
            self.key = f"ta:{self.tA},tb:{self.tB},m:{self.m},n{self.n},k{self.k}"
        elif match := re.match(
            self.STRIDED_BATCHED_ROCBLAS_BENCH_RE, rocblas_bench_string
        ):
            self.match = match
            self.gemm_type = "Strided batched"
            self.count = 1
            self.tA = self.match.group("TRANSPOSE_A")
            self.tB = self.match.group("TRANSPOSE_B")
            self.m = int(self.match.group("M"))
            self.n = int(self.match.group("N"))
            self.k = int(self.match.group("K"))
            self.alpha = float(self.match.group("ALPHA"))
            self.lda = int(self.match.group("LDA"))
            self.ldb = int(self.match.group("LDB"))
            self.beta = float(self.match.group("BETA"))
            self.ldc = int(self.match.group("LDC"))
            self.compute_type = self.match.group("COMPUTE_TYPE")
            self.a_type = self.match.group("A_TYPE")
            self.stride_a = int(self.match.group("STRIDE_A"))
            self.stride_b = int(self.match.group("STRIDE_B"))
            self.stride_c = int(self.match.group("STRIDE_C"))
            self.stride_d = int(self.match.group("STRIDE_D"))
            self.batch_count = int(self.match.group("BATCH_COUNT"))
            self.key = f"ta:{self.tA},tb:{self.tB},m:{self.m},n{self.n},k{self.k},sa:{self.stride_a},sb:{self.stride_b},sc:{self.stride_c},bc:{self.batch_count}"
        else:
            self.match = False

    def __bool__(self):
        return True if self.match else False

    def inc_count(self, number=1):
        self.count += number

    def __repr__(self):
        return f"Instances: {self.count} M: {self.m} n: {self.n} k: {self.k}"

    def run_args(self):
        if self.gemm_type == "Generic":
            return self.tA, self.tB, self.m, self.n, self.k, self.alpha, self.beta
        elif self.gemm_type == "Strided batched":
            return (
                self.tA,
                self.tB,
                self.m,
                self.n,
                self.k,
                self.alpha,
                self.beta,
                self.stride_a,
                self.stride_b,
                self.stride_c,
                self.batch_count,
            )

    def csv_list(self):
        # Only two possible formats? from snooping: UserDrivenTuningParser.cpp in tensile
        if self.gemm_type == "Generic":
            return [
                self.tA,
                self.tB,
                self.m,
                self.n,
                1,
                self.k,
                self.alpha,
                self.beta,
                self.lda,
                self.ldb,
                self.ldc,
                self.a_type,
                self.a_type,
                self.compute_type,
                self.solution_index,
            ]
        else:
            return [
                self.tA,
                self.tB,
                self.m,
                self.n,
                self.batch_count,
                self.k,
                self.alpha,
                self.beta,
                self.lda,
                self.ldb,
                self.ldc,
                self.stride_a,
                self.stride_b,
                self.stride_c,
                self.a_type,
                self.a_type,
                self.compute_type,
                self.solution_index,
            ]


class ExecutableRunner:
    """
    Class for running any executable, with the correct env variable, and collect logs
    """

    def __init__(self, executable):
        self.executable = executable

    def run_and_collect(self, show_output=False):
        env = os.environ.copy()
        env["ROCBLAS_LAYER"] = "2"
        # TODO: Needs a "try catch"
        process = subprocess.run(
            self.executable, stderr=subprocess.PIPE, text=True, env=env
        )
        self.process_output = process.stderr
        if show_output:
            print(f"Output from subprocess.run: {self.process_output}")

    def get_unique_gemms(self):
        """
        Return every unique gemm with the form [Count, TransposeA, TransposeB, M, N, K]
        """
        out_dict = {}
        lines = self.process_output.splitlines()
        for line in lines:
            if gemm := GEMM(line):
                # TODO Seems like there should be a better way?
                if gemm.key in out_dict:
                    out_dict[gemm.key].inc_count
                else:
                    out_dict[gemm.key] = gemm
        return list(out_dict.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        help="Output file with the results. NOT IMPLEMENTED YET",
        action="store",
        dest="output",
        default="BlasterOutput.csv",
    )
    parser.add_argument("--show_gemms", action="store_true")
    parser.add_argument("executable", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # Run and collect
    executable = ExecutableRunner(args.executable)
    executable.run_and_collect()
    print(f"{os.linesep}{'>'*20:<20}{' rocBlas Output ':^20}{'<'*20:>20}{os.linesep}")
    gemms = executable.get_unique_gemms()
    if args.show_gemms:
        print(f"Got unique gemms {gemms}")

    tunner = rocBlasFinder()
    total_old = 0
    total_new = 0
    for gemm in gemms:
        # TODO: Best to pass a list?
        results = tunner.run(*gemm.run_args())
        # TODO: Check if bad?
        match = re.match(
            r"Default: (\d+.\d+) Winner: (\d+.\d+) Solution: (\d+)", results
        )
        default_time = float(match.group(1))
        winning_time = float(match.group(2))
        solution_nu = int(match.group(3))
        print(f"Improved by: {(default_time-winning_time)/default_time}{os.linesep}")
        total_old += int(gemm.count) * default_time
        total_new += int(gemm.count) * winning_time
        # Write new solution to gemm
        gemm.solution_index = solution_nu
    print(
        f"{os.linesep}{'>'*20:<20}{' Summary ':^20}{'<'*20:>20}{os.linesep}"
        f"Old time: {total_old}{os.linesep}"
        f"New time: {total_new}{os.linesep}"
        f"Total improvement: {(total_old-total_new)/total_old:0.2f}"
    )

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        headers = [
            "transA",
            "transB",
            "M",
            "N",
            "batch_count",
            "K",
            "alpha",
            "beta",
            "lda",
            "ldb",
            "ldc",
            "input_type",
            "output_type",
            "compute_type",
            "solution_index",
        ]
        writer.writerow(headers)
        for gemm in gemms:
            writer.writerow(gemm.csv_list())


if __name__ == "__main__":
    main()
