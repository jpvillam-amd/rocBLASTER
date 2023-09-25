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
        self.match = re.match(self.GENERIC_ROCBLAS_BENCH_RE, rocblas_bench_string)
        self.rocblas_bench_string = rocblas_bench_string
        if self.match:
            self.count = 1
            self.tA = self.match.group("TRANSPOSE_A")
            self.tB = self.match.group("TRANSPOSE_B")
            self.m = int(self.match.group("M"))
            self.n = int(self.match.group("N"))
            self.k = int(self.match.group("K"))
            self.alpha = int(self.match.group("ALPHA"))
            self.lda = int(self.match.group("LDA"))
            self.ldb = int(self.match.group("LDB"))
            self.beta = int(self.match.group("BETA"))
            self.ldc = int(self.match.group("LDC"))
            self.compute_type = self.match.group("COMPUTE_TYPE")
            self.input_type = self.match.group("A_TYPE")
            self.output_type = self.match.group("C_TYPE")
            self.key = f"ta:{self.tA},tb:{self.tB},m:{self.m},n{self.n},k{self.k}"
            self.solution_index = int(self.match.group("SOLUTION_INDEX"))

    def __bool__(self):
        return True if self.match else False

    def inc_count(self, number=1):
        self.count += number

    def __repr__(self):
        return f"Instances: {self.count} M: {self.m} N: {self.n} K: {self.k} solution_index: {self.solution_index}\n"

    def csv_list(self):
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
            self.input_type,
            self.output_type,
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
                    out_dict[gemm.key].inc_count()
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
        results = tunner.run(gemm.tA, gemm.tB, gemm.m, gemm.n, gemm.k)
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
