import numpy as np

# Grid Configuration
GRID_WIDTH = 800  # Original WIDTH
GRID_HEIGHT = 600  # Original HEIGHT
TORUS_SPACE = True

# Patch system for odor diffusion
PATCH_WIDTH = 100
PATCH_HEIGHT = 75
SCALE = PATCH_WIDTH / GRID_WIDTH

# Initialize odor patches (4 channels)
PATCHES = [np.zeros((PATCH_WIDTH, PATCH_HEIGHT)) for _ in range(4)]

# Simulation Settings
FPS = 30
MAX_STEPS = 1000

# Agent Populations
HERMI_POPULATION_DEFAULT = 4
FLAB_POPULATION_DEFAULT = 4
FAUXFLAB_POPULATION_DEFAULT = 4

# Learning Parameters (exact from original)
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
SENSOR_DISTANCE = 5
ENCOUNTER_COOLDOWN = 10
PREY_RADIUS = 5
NUM_ODOR_TYPES = 4

# Debug
DEBUG_MODE = False

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BROWN = (139, 69, 19)
RED = (255, 0, 0)
CYAN = (0, 255, 255)
PINK = (255, 192, 203)
YELLOW = (255, 255, 0)
