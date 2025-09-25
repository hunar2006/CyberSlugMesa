import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from visualization import page
import time

if __name__ == "__main__":
    print("Cyberslug simulation starting...")
    print("If browser doesn't open automatically, try:")
    print("http://localhost:8765")
    print("http://localhost:8521")

    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Simulation stopped.")

