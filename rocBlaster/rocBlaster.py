#!/usr/bin/env python
"""
Simple programs that gets all BLAS GEMMs from an executable and tries to tune them.
"""
import argparse
import subprocess
import os
import re

# TODO: Need to figure out this relative path
from rocBlaster.rocBlasFinder import rocBlasFinder


class ExecutableRunner:
    """
    Class for running any executable, with the correct env variable, and collect logs
    """

    # Constants
    GENERIC_ROCBLAS_BENCH_RE = r"./rocblas-bench -f gemm -r (\w+) --transposeA (\w) --transposeB (\w) -m (\w+) -n (\w+) -k (\w+) --alpha (\w+) --lda (\w+) --ldb (\w+) --beta (\w+) --ldc (\w+)"
    TRANSPOSE_A = 2
    TRANSPOSE_B = 3
    M = 4
    N = 5
    K = 6

    def __init__(self, executable):
        self.executable = executable

    def run_and_collect(self):
        env = os.environ.copy()
        env["ROCBLAS_LAYER"] = "2"
        # TODO: Needs a "try catch"
        process = subprocess.run(
            self.executable, capture_output=True, text=True, env=env
        )
        self.process_output = process.stderr
        print(f"Output from subprocess.run: {self.process_output}")

    def get_unique_gemms(self):
        """
        Return every unique gemm with the form [Count, TransposeA, TransposeB, M, N, K]
        """
        out_dict = {}
        lines = self.process_output.splitlines()
        for line in lines:
            if match := re.match(self.GENERIC_ROCBLAS_BENCH_RE, line):
                tA = match.group(self.TRANSPOSE_A)
                tB = match.group(self.TRANSPOSE_B)
                m = match.group(self.M)
                n = match.group(self.N)
                k = match.group(self.K)
                key = f"ta:{tA},tb:{tB},m:{m},n{n},k{k}"
                # TODO Seems like there should be a better way, maybe a custom class?
                if entry := out_dict.get(key, [0, tA, tB, m, n, k]):
                    entry[0] += 1
                    out_dict[key] = entry
        return list(out_dict.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        help="Output file with the results.",
        action="store",
        dest="output",
        default="BlaterOutput.txt",
    )
    parser.add_argument("executable", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # Run and collect
    executable = ExecutableRunner(args.executable)
    executable.run_and_collect()
    gemms = executable.get_unique_gemms()
    print(f"Got unique gemms {gemms}")

    tunner = rocBlasFinder()
    total_old = 0
    total_new = 0
    for gemm in gemms:
        # TODO: Best to pass a list - first element?
        results = tunner.run(gemm[1], gemm[2], int(gemm[3]), int(gemm[4]), int(gemm[5]))
        # TODO: Check if bad?
        match = re.match(
            r"Default: (\d+.\d+) Winner: (\d+.\d+) Solution: (\d+)", results
        )
        default_time = float(match.group(1))
        winning_time = float(match.group(2))
        solution_nu = int(match.group(3))
        print(f"Improved by: {(default_time-winning_time)/default_time}")
        total_old += int(gemm[0]) * default_time
        total_new += int(gemm[0]) * winning_time
        # TODO Write solutions out file
    print(
        f"Old time: {total_old}\nNew time: {total_new}\nTotal improvement: {(total_old-total_new)/total_old:0.2f}"
    )


if __name__ == "__main__":
    main()
