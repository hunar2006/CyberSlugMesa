import numpy as np

# Mesa Configuration
GRID_WIDTH = 60  # Mesa grid cells
GRID_HEIGHT = 60
TORUS_SPACE = True  # Wrap-around edges

# Simulation Settings
FPS = 10  # Slower for scientific observation
MAX_STEPS = 1000

# Agent Populations
CYBERSLUG_COUNT = 1
HERMI_POPULATION = 4
FLAB_POPULATION = 4
FAUXFLAB_POPULATION = 4

# Learning Parameters
ALPHA_HERMI = 0.5
BETA_HERMI = 1.0
LAMBDA_HERMI = 1.0
ALPHA_FLAB = 0.5
BETA_FLAB = 1.0
LAMBDA_FLAB = 1.0
ALPHA_DRUG = 0.5
BETA_DRUG = 1.0
LAMBDA_DRUG = 1.0

# Odor Signatures (4-channel system)
FLAB_ODOR = [0.5, 0, 0.5, 0]
HERMI_ODOR = [0.5, 0.5, 0, 0]
DRUG_ODOR = [0, 0, 0, 0.5]

# Sensing Parameters
SENSOR_DISTANCE = 2  # Grid cells
ENCOUNTER_COOLDOWN = 10
PREY_RADIUS = 1

# Visualization Colors
COLORS = {
    'cyberslug': '#8B4513',  # Brown
    'hermi': '#00FFFF',      # Cyan
    'flab': '#FF69B4',       # Pink
    'fauxflab': '#FFFF00',   # Yellow
    'path': '#000000'        # Black
}
