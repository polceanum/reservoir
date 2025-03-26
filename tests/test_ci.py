import unittest
import sys
import io
import os
import time
from contextlib import redirect_stdout

# Adjust the module search path to include the ../src folder.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import sparse_reservoir as sr

class TestReservoir(unittest.TestCase):
    def _run_experiment(self, test_args, expected_threshold):
        sys.argv = test_args
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            sr.main()
        output = captured_output.getvalue()
        mse = None
        for line in output.splitlines():
            if line.startswith("MSE ="):
                try:
                    mse = float(line.split("MSE =")[1].strip())
                except ValueError:
                    pass
        self.assertIsNotNone(mse, "MSE value not found in experiment output")
        self.assertLess(mse, expected_threshold, f"Model error {mse} is too high, regression detected.")
        print(f"Test completed: MSE = {mse:.8f} (< {expected_threshold}).")
        return mse

    def test_error_linear_solver(self):
        test_args = [
            "sparse_reservoir.py",
            "--data-file", "./data/MackeyGlass_t17.txt",
            "--fp", "64",
            "--opt", "lr",
            "--top", "uniform",
            "--dim-res", "1000",
            "--rho", "0.01",        # reservoir density
            "--alpha", "0.3",       # reservoir leak rate
            "--rest",               # enable spectral radius estimation
            "--read-out", "linear", # linear read-out architecture
            "--valve-in", "1000",   # input valve size
            "--valve-out", "1000",  # output valve size
            "--dim-in", "1",
            "--dim-out", "1",
            "--device", "cpu"
        ]
        expected_threshold = 1e-6
        self._run_experiment(test_args, expected_threshold)

    def test_error_gradient_descent(self):
        test_args = [
            "sparse_reservoir.py",
            "--data-file", "./data/MackeyGlass_t17.txt",
            "--fp", "32",
            "--lr", "0.01",
            "--opt", "adam",
            "--epochs", "1000",
            "--top", "uniform",
            "--dim-res", "1000",
            "--rho", "0.01",
            "--alpha", "0.3",
            "--rest",
            "--read-out", "linear",
            "--valve-in", "1000",
            "--valve-out", "1000",
            "--dim-in", "1",
            "--dim-out", "1",
            "--device", "cpu"
        ]
        expected_threshold = 0.02
        self._run_experiment(test_args, expected_threshold)

    def test_error_llm_input(self):
        test_args = [
            "sparse_reservoir.py",
            "--fp", "32",
            "--lr", "0.01",
            "--opt", "adam",
            "--epochs", "1000",
            "--top", "uniform",
            "--dim-res", "1000",
            "--rho", "0.01",
            "--alpha", "0.3",
            "--rest",
            "--read-out", "linear",
            "--valve-in", "1000",
            "--valve-out", "1000",
            "--device", "cpu",
            "--dataset", "wikipedia"
        ]
        expected_threshold = 0.3
        self._run_experiment(test_args, expected_threshold)

    def test_runtime_regression(self):
        test_args = [
            "sparse_reservoir.py",
            "--data-file", "./data/MackeyGlass_t17.txt",
            "--fp", "32",
            "--lr", "0.01",
            "--opt", "adam",
            "--epochs", "1000",
            "--top", "uniform",
            "--dim-res", "1000",
            "--rho", "0.01",
            "--alpha", "0.3",
            "--rest",
            "--read-out", "linear",
            "--valve-in", "1000",
            "--valve-out", "1000",
            "--dim-in", "1",
            "--dim-out", "1",
            "--device", "cpu"
        ]
        runtime_threshold = 5.0  # seconds
        sys.argv = test_args
        start_time = time.perf_counter()
        sr.main()
        elapsed = time.perf_counter() - start_time
        self.assertLess(elapsed, runtime_threshold, f"Runtime {elapsed:.2f}s exceeds threshold of {runtime_threshold}s.")
        print(f"Test runtime_regression: Runtime = {elapsed:.2f}s (< {runtime_threshold}s).")

if __name__ == '__main__':
    unittest.main()
