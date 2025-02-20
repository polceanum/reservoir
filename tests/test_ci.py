import unittest
import sys
import io
import os
from contextlib import redirect_stdout

# Adjust the module search path to include the ../src folder.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import sparse_reservoir as sr

class TestModelErrorRegression(unittest.TestCase):
    def _run_experiment(self, test_args, expected_threshold):
        """
        Helper method that:
          - Sets sys.argv to the provided test_args.
          - Runs sr.main() and captures its stdout.
          - Parses the output for a line starting with "MSE =" to get the error.
          - Asserts that the MSE exists and is below expected_threshold.
          - Prints a confirmation message that includes the test name.
          - Returns the parsed MSE.
        """
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
        
        # Get test name from self.id() for a generic confirmation message.
        test_name = self.id().split('.')[-1]
        print(f"Test {test_name}: MSE = {mse:.8f}, which is within the expected range (< {expected_threshold}).")
        return mse

    def test_linear_solver(self):
        test_args = [
            "sparse_reservoir.py",
            "--data-file", "./data/MackeyGlass_t17.txt",
            "--fp", "64",
            "--opt", "lr",
            "--top", "uniform",
            "--dim-res", "1000",
            "--rho", "0.01",       # reservoir density
            "--alpha", "0.3",      # reservoir leak rate
            "--rest",             # enable reservoir spectral radius estimation
            "--read-out", "linear",# linear read-out architecture
            "--valve-in", "1000",  # input valve size
            "--valve-out", "1000", # output valve size
            "--dim-in", "1",
            "--dim-out", "1",
            "--device", "cpu"
        ]
        expected_threshold = 1e-6
        self._run_experiment(test_args, expected_threshold)

    def test_gradient_descent(self):
        test_args = [
            "sparse_reservoir.py",
            "--data-file", "./data/MackeyGlass_t17.txt",
            "--fp", "32",
            "--lr", "0.01",
            "--opt", "adam",
            "--epochs", "1000",
            "--top", "uniform",
            "--dim-res", "1000",
            "--rho", "0.01",       # reservoir density
            "--alpha", "0.3",      # reservoir leak rate
            "--rest",             # enable reservoir spectral radius estimation
            "--read-out", "linear",# linear read-out architecture
            "--valve-in", "1000",  # input valve size
            "--valve-out", "1000", # output valve size
            "--dim-in", "1",
            "--dim-out", "1",
            "--device", "cpu"
        ]
        expected_threshold = 0.02
        self._run_experiment(test_args, expected_threshold)

if __name__ == '__main__':
    unittest.main()
