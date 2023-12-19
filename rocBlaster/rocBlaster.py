#!/usr/bin/env python
"""
Simple programs that gets all BLAS GEMMs from an executable and tries to tune them.
"""
import argparse
import subprocess
import os
import re
import csv
import mimetypes
from multiprocessing import Process, Queue, set_start_method
import signal
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
        # First match the gemm
        self.rocblas_bench_string = rocblas_bench_string
        if match := re.match(self.GENERIC_ROCBLAS_BENCH_RE, rocblas_bench_string):
            self.match = True
            self.gemm_type = "Generic"
        elif match := re.match(
            self.STRIDED_BATCHED_ROCBLAS_BENCH_RE, rocblas_bench_string
        ):
            self.match = True
            self.gemm_type = "Strided batched"
        else:
            self.match = False

        # Collect data in new if so we can share code
        if self.match:
            self.count = 1
            self.tA = match.group("TRANSPOSE_A")
            self.tB = match.group("TRANSPOSE_B")
            self.m = int(match.group("M"))
            self.n = int(match.group("N"))
            self.k = int(match.group("K"))
            self.alpha = float(match.group("ALPHA"))
            self.lda = int(match.group("LDA"))
            self.ldb = int(match.group("LDB"))
            self.beta = float(match.group("BETA"))
            self.ldc = int(match.group("LDC"))
            self.compute_type = match.group("COMPUTE_TYPE")
            self.a_type = match.group("A_TYPE")
            self.solution_index = match.group("SOLUTION_INDEX")
            if self.gemm_type == "Generic":
                self.key = f"ta:{self.tA},tb:{self.tB},m:{self.m},n{self.n},k{self.k}"
            elif self.gemm_type == "Strided batched":
                self.stride_a = int(match.group("STRIDE_A"))
                self.stride_b = int(match.group("STRIDE_B"))
                self.stride_c = int(match.group("STRIDE_C"))
                self.stride_d = int(match.group("STRIDE_D"))
                self.batch_count = int(match.group("BATCH_COUNT"))
                self.key = f"ta:{self.tA},tb:{self.tB},m:{self.m},n{self.n},k{self.k},sa:{self.stride_a},sb:{self.stride_b},sc:{self.stride_c},bc:{self.batch_count}"

    def __bool__(self):
        return self.match

    def inc_count(self, number=1):
        self.count += number

    def __repr__(self):
        return f"Instances: {self.count} M: {self.m} N: {self.n} K: {self.k} solution_index: {self.solution_index}\n"

    def run_args(self):
        if self.gemm_type == "Generic":
            return self.tA, self.tB, self.m, self.n, self.k, self.alpha, self.beta, self.a_type, self.a_type
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
                self.a_type,
                self.a_type
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
        env["ROCBLAS_LOG_BENCH_PATH"] = env.get("ROCBLAS_LOG_BENCH_PATH", "/tmp/rocblas_bench_log.txt")
        # TODO: Needs to swap to "4" and read csv
        process = subprocess.run(
            self.executable,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        # self.process_output = process.stdout
        if show_output:
            print(f"Output from subprocess.run: {process.stdout}")
        return env["ROCBLAS_LOG_BENCH_PATH"]

def handler(signum, frame):
    raise Exception("time out")

def run_tuning(gpu_id, in_q, out_q, timeout):
    tunner = rocBlasFinder(gpu_id)
    while in_q.qsize():
        gemm = in_q.get()
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout)

        try:
            results = tunner.run(*gemm.run_args())
        except Exception as exc:
            signal.alarm(0)
            print('\n', gemm, exc)
        else:
            signal.alarm(0)
            # TODO: Check if bad?
            match = re.match(
                r"Default: (\d+.\d+) Winner: (\d+.\d+) Solution: (\d+)", results
            )
            default_time = float(match.group(1))
            winning_time = float(match.group(2))
            solution_nu = int(match.group(3))
            old_time = int(gemm.count) * default_time
            new_time = int(gemm.count) * winning_time
            # Write new solution to gemm
            gemm.solution_index = solution_nu
            if new_time<old_time:
                out_q.put((gemm, old_time, new_time))


def process_gemms(gemms, timeout):
    gpu_ids = [int(gpu_id) for gpu_id in os.environ.get('HIP_VISIBLE_DEVICES', '0').split(',')]
    in_q = Queue()
    out_q = Queue()

    for gemm in gemms:
        in_q.put(gemm)

    processes = []
    for gpu_id in range(len(gpu_ids)):
        p = Process(target=run_tuning, args=(gpu_id, in_q, out_q, timeout))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
        p.close()

    total_old = 0
    total_new = 0
    gemms = []
    while out_q.qsize():
        gemm, old_time, new_time = out_q.get()
        gemms.append(gemm)
        total_old += old_time
        total_new += new_time
    return gemms, total_old, total_new


def main():
    set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        help="Output file with the results.",
        action="store",
        dest="output",
        default="BlasterOutput.csv",
    )
    parser.add_argument("--show_gemms", action="store_true")
    parser.add_argument("--timeout", default=60, type=int, help="Gemm tuning timeout(seconds).")
    parser.add_argument("--show_output", action="store_true")
    parser.add_argument("executable", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    mime = mimetypes.guess_type(args.executable[0])
    
    if mime[0] == 'text/plain':
        # Work with ROCBLAS_LOG_BENCH_PATH
        rocblas_bench_log = args.executable[0]
    else:
        # Run and collect
        executable = ExecutableRunner(args.executable)
        rocblas_bench_log = executable.run_and_collect(args.show_output)
        print(f"{os.linesep}{'>'*20:<20}{' rocBlas Output ':^20}{'<'*20:>20}{os.linesep}")

    out_dict = {}
    with open(rocblas_bench_log, 'r') as f:
        for line in f.readlines():
            if gemm := GEMM(line):
                # TODO Seems like there should be a better way?
                if gemm.key in out_dict:
                    out_dict[gemm.key].inc_count()
                else:
                    out_dict[gemm.key] = gemm
        gemms = list(out_dict.values())

    if args.show_gemms:
        print(f"Got unique gemms {gemms}")

    gemms, total_old, total_new = process_gemms(gemms, args.timeout)
    
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
