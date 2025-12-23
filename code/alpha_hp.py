import matplotlib.pyplot as plt
import numpy as np
from ap_utils import (
    APSignal, evaluate_model, to_log
)

# --------------------------- CONFIG --------------------------- #
DATA_FILE = "data/min0_all.npy"       # Path to data
RESULTS_FILE = "results/alpha_results.txt"    # Output log file
MINUTE = 0                             # Minute selection
TOLERANCE = 20                         # Sampling interval factor
VOLTAGE = -1                           # Voltage
THRESHOLDS = {0: -84, 3: -74, 5: -67, 9: -65, 13: -66}  

# Model/Training defaults
DEFAULT_LBACK = 32
DEFAULT_HP = 8
DEFAULT_EPOCHS = 1000  #uses early stopping
DEFAULT_BATCH = 32
DEFAULT_NCLUSTERS = 10

# Flags for plotting
PLOT_LOSS = False
PLOT_ACCURACY = False
PLOT_PULSES = True

# Default model
DEFAULT_MODEL = 'reg'  # Options: 'reg', 'lstm', 'gru', 'cnn'


# ------------------------ UTIL FUNCTIONS ------------------------ #
def init_ap_signal():
    """Initialize APSignal object."""
    return APSignal(
        DATA_FILE,
        t_tol=TOLERANCE,
        v_tol=VOLTAGE,
        threshold=THRESHOLDS[MINUTE],
        show=False,
        mode=1,
        normalize=False
    )


def log_header(fout):
    """Write a standard header to the log file."""
    to_log('------------------------------------------', fout)


def run_evaluation(ap, fout):
    """Evaluate a single model using the new evaluate_model function."""
    return evaluate_model(
        ap,
        fout,
        mode=DEFAULT_MODEL,
        lback=DEFAULT_LBACK,
        hp=DEFAULT_HP,
        epochs=DEFAULT_EPOCHS,
        batch_size=DEFAULT_BATCH,
        nclusters=DEFAULT_NCLUSTERS,
        sequential=False  # Keep standard horizon evaluation as before
    )


# ------------------------ MAIN SCRIPT ------------------------ #
if __name__ == "__main__":
    with open(RESULTS_FILE, "a") as fout:
        log_header(fout)
        
        # Initialize signal
        ap = init_ap_signal()
        
        # Run evaluation using the new evaluate_model
        results = run_evaluation(ap, fout)
        
        print("Single Model Evaluation:")
        print(f"Model Type: {results['model_type']}")
        print(f"Prediction Error: {results['error']}")
        print(f"Overall Error: {results.get('overall_error', 'N/A')}")
