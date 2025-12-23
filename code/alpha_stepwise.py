import matplotlib.pyplot as plt
import numpy as np
from ap_utils import (
    APSignal, evaluate_model, to_log
)

# --------------------------- CONFIG --------------------------- #
# General parameters
MINUTE = 0
TOLERANCE = 20       # dt = 0.05, 1 of every 20 samples
VOLTAGE = -1
DATA_PATH = "data/min0_all.npy"

# Logging
RESULT_FILE = "results/stepwise_results.txt"
fout = open(RESULT_FILE, "a")
to_log('------------------------------------------', fout)

# Thresholds per minute
THRESHOLDS = {0: -84, 3: -74, 5: -67, 9: -65, 13: -66}

# Model parameters
LOOKBACK = 64
HORIZON = 8
EPOCHS = 1000
BATCH_SIZE = 32
NCLUSTERS = 10
STEPSIZE = 2
PLOT_LOSS = False
PLOT_ACCURACY = False
PLOT_PULSES = True

# Default model
DEFAULT_MODEL = 'reg'  # Options: 'reg', 'lstm', 'gru', 'cnn'

# ------------------------ FUNCTIONS ------------------------ #
def init_ap_signal(minute=MINUTE, mode=0, show=False, normalize=False):
    """Initialize the APSignal object."""
    return APSignal(
        DATA_PATH,
        t_tol=TOLERANCE,
        v_tol=VOLTAGE,
        threshold=THRESHOLDS[minute],
        show=show,
        mode=mode,
        normalize=normalize
    )


def run_stepwise_model(ap, fout, mode=DEFAULT_MODEL, lookback=LOOKBACK, hp=HORIZON, steps=STEPSIZE):
    """Run stepwise prediction using the new evaluate_model function."""
    return evaluate_model(
        ap,
        fout,
        mode=mode,
        lback=lookback,
        hp=hp,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        nclusters=NCLUSTERS,
        sequential=True,  # Use sequential (stepwise) prediction
        steps=steps,
        plot_loss=PLOT_LOSS,
        plot_accuracy=PLOT_ACCURACY,
        plot_pulses=PLOT_PULSES
    )


# ------------------------ MAIN SCRIPT ------------------------ #
if __name__ == "__main__":
    # Initialize AP signal
    ap = init_ap_signal(mode=0, show=False, normalize=False)
    
    # Example: Stepwise prediction
    results = run_stepwise_model(ap, fout, mode='reg', hp=HORIZON, steps=1)
    
    # Print results
    print("Stepwise Model Evaluation:")
    for key, value in results.items():
        print(f"{key}: {value}")

    fout.close()
