import numpy as np

# Grid Configuration - SMALLER GRID for Mesa compatibility
GRID_WIDTH = 60  # Much smaller for Mesa
GRID_HEIGHT = 60  # Square grid
TORUS_SPACE = True

# Patch system for odor diffusion - ADJUSTED
PATCH_WIDTH = 30   # Half of grid width
PATCH_HEIGHT = 30  # Half of grid height
SCALE = PATCH_WIDTH / GRID_WIDTH  # This becomes 0.5

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

# Odor Signatures (4-channel system) - STRONGER ODORS
FLAB_ODOR = [1.0, 0, 1.0, 0]      # 2x stronger
HERMI_ODOR = [1.0, 1.0, 0, 0]     # 2x stronger
DRUG_ODOR = [0, 0, 0, 1.0]        # 2x stronger

# Sensing Parameters
SENSOR_DISTANCE = 2  # Smaller distance for smaller grid
ENCOUNTER_COOLDOWN = 10
PREY_RADIUS = 5
NUM_ODOR_TYPES = 4

# Debug - TURN ON to see what's happening
DEBUG_MODE = True  # Turn this on!

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BROWN = (139, 69, 19)
RED = (255, 0, 0)
CYAN = (0, 255, 255)
PINK = (255, 192, 203)
YELLOW = (255, 255, 0)
