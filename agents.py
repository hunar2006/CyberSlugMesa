from mesa import Agent
import math
import random
from config import *


def wrap_around(x, y):
    """Wrap coordinates around grid boundaries"""
    return x % GRID_WIDTH, y % GRID_HEIGHT


class CyberslugAgent(Agent):
    """EXACT translation of original Cyberslug class from sluggame.py"""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        # EXACT COPY from original __init__
        self.x, self.y = GRID_WIDTH // 2, GRID_HEIGHT // 2
        self.angle = 0  # degrees
        self.speed = 3
        self.path = [(self.x, self.y)]

        # Learning and motivation variables - EXACT COPY
        self.nutrition = 0.5
        self.incentive = 0.0
        self.satiation = 0.0
        self.app_state = 0.0
        self.app_state_switch = 0.0

        # Somatic map - EXACT COPY
        self.somatic_map = 0.0

        # Learning variables - EXACT COPY
        self.Vh, self.Vf, self.Vd = 0.0, 0.0, 0.0
        self.alpha_hermi, self.beta_hermi, self.lambda_hermi = ALPHA_HERMI, BETA_HERMI, LAMBDA_HERMI
        self.alpha_flab, self.beta_flab, self.lambda_flab = ALPHA_FLAB, BETA_FLAB, LAMBDA_FLAB
        self.alpha_drug, self.beta_drug, self.lambda_drug = ALPHA_DRUG, BETA_DRUG, LAMBDA_DRUG

        # Sensor arrays - EXACT COPY
        self.sns_odors_left = [0.0] * NUM_ODOR_TYPES
        self.sns_odors_right = [0.0] * NUM_ODOR_TYPES
        self.sns_odors = [0.0] * NUM_ODOR_TYPES

        # Pain and reward mechanisms - EXACT COPY
        self.sns_pain_left = 0.0
        self.sns_pain_right = 0.0
        self.spontaneous_pain = 2.0
        self.reward_experience = 0.0  # THIS WAS THE MISSING CRITICAL VARIABLE!

        # Counters and timers - EXACT COPY
        self.encounter_timer = 0
        self.hermi_counter, self.flab_counter, self.drug_counter = 0, 0, 0

        # Additional variables from original update method
        self.sns_pain = 0.0
        self.pain = 0.0
        self.pain_switch = 0.0
        self.reward_pos = 0.0
        self.reward_neg = 0.0
        self.somatic_map_senses_left = []
        self.somatic_map_senses_right = []
        self.somatic_map_senses = []
        self.somatic_map_factors = []
        self.somatic_map_sigmoids = []
        self.turn_angle = 0.0

        # Set initial position for Mesa
        self.pos = (self.x, self.y)

    def step(self):
        """Mesa step method - calls original update logic"""
        sensors_left, sensors_right = self.get_sensor_readings()
        encounter = self.check_encounters()

        # Call EXACT original update method
        turn_angle = self.update(sensors_left, sensors_right, encounter)

        # Move based on turn angle
        self.move_cyberslug(turn_angle)

    def get_sensor_readings(self):
        """Get sensor readings using EXACT original sensor logic"""
        if not self.pos:
            return [0.0] * NUM_ODOR_TYPES, [0.0] * NUM_ODOR_TYPES

        # Update internal x,y from Mesa pos
        self.x, self.y = self.pos

        # EXACT COPY of original sensors function from utils.py
        px, py = self.convert_patch_to_coord(self.x, self.y)

        left_x = int(px + SENSOR_DISTANCE * math.cos(math.radians(self.angle + 45)))
        left_y = int(py + SENSOR_DISTANCE * math.sin(math.radians(self.angle + 45)))
        right_x = int(px + SENSOR_DISTANCE * math.cos(math.radians(self.angle - 45)))
        right_y = int(py + SENSOR_DISTANCE * math.sin(math.radians(self.angle - 45)))

        left_x, left_y = max(0, min(PATCH_WIDTH - 1, left_x)), max(0, min(PATCH_HEIGHT - 1, left_y))
        right_x, right_y = max(0, min(PATCH_WIDTH - 1, right_x)), max(0, min(PATCH_HEIGHT - 1, right_y))

        # Get odor readings from model patches
        left_odors = [self.model.patches[i][left_x, left_y] for i in range(NUM_ODOR_TYPES)]
        right_odors = [self.model.patches[i][right_x, right_y] for i in range(NUM_ODOR_TYPES)]

        return left_odors, right_odors

    def convert_patch_to_coord(self, x, y):
        """EXACT COPY from original utils.py"""
        px = int((x - GRID_WIDTH / 2) * SCALE + PATCH_WIDTH / 2)
        py = int((y - GRID_HEIGHT / 2) * SCALE + PATCH_HEIGHT / 2)
        px = max(0, min(PATCH_WIDTH - 1, px))
        py = max(0, min(PATCH_HEIGHT - 1, py))
        return px, py

    def check_encounters(self):
        """Check for encounters with prey"""
        if self.encounter_timer > 0:
            return "none"

        cellmates = self.model.grid.get_cell_list_contents([self.pos])

        for agent in cellmates:
            if isinstance(agent, PreyAgent) and agent != self:
                encounter_type = agent.prey_type
                agent.respawn()
                return encounter_type

        return "none"

    def update(self, sensors_left, sensors_right, encounter):
        """EXACT COPY of original update method from sluggame.py"""

        # EXACT COPY - sensor processing
        self.sns_odors_left = [0 if i <= 1e-7 else (7 + math.log10(i)) for i in sensors_left]
        self.sns_odors_right = [0 if i <= 1e-7 else (7 + math.log10(i)) for i in sensors_right]
        self.sns_odors = [(l + r) / 2 for l, r in zip(self.sns_odors_left, self.sns_odors_right)]

        sns_betaine, sns_hermi, sns_flab, sns_drug = self.sns_odors

        # --- Associative learning from prey encounters --- EXACT COPY
        if encounter == "hermi":
            self.Vh += self.alpha_hermi * self.beta_hermi * (self.lambda_hermi - self.Vh)
            self.nutrition += 0.1
            if self.encounter_timer == 0:
                self.hermi_counter += 1
                self.encounter_timer = ENCOUNTER_COOLDOWN

        if encounter == "flab":
            self.Vf += self.alpha_flab * self.beta_flab * (self.lambda_flab - self.Vf)
            self.nutrition += 0.1
            if self.encounter_timer == 0:
                self.flab_counter += 1
                self.encounter_timer = ENCOUNTER_COOLDOWN

        if encounter == "drug":
            self.Vd += self.alpha_drug * self.beta_drug * (self.lambda_drug - self.Vd)
            if self.encounter_timer == 0:
                self.drug_counter += 1
                self.encounter_timer = ENCOUNTER_COOLDOWN

        # --- Pain Calculations --- EXACT COPY
        self.sns_pain = (self.sns_pain_left + self.sns_pain_right) / 2
        self.pain = 10 / (1 + math.exp(-2 * (self.sns_pain + self.spontaneous_pain) + 10))
        self.pain_switch = 1 - 2 / (1 + math.exp(-10 * (self.sns_pain - 0.2)))

        # --- Nutrition, Satiation, and Incentive Calculations --- EXACT COPY
        self.nutrition -= 0.005 * self.nutrition
        self.satiation = 1 / ((1 + 0.7 * math.exp(-4 * self.nutrition + 2)) ** 2)

        self.reward_pos = (
                sns_betaine / (1 + (0.05 * self.Vh * sns_hermi) - 0.006 / self.satiation)
                + 3.0 * self.Vh * sns_hermi
                + 8.0 * self.Vd * sns_drug)

        self.reward_neg = 0.59 * self.Vf * sns_flab
        self.incentive = self.reward_pos - self.reward_neg

        # --- Somatic Map Calculation --- EXACT COPY
        self.somatic_map_senses_left = self.sns_odors_left[1:] + [self.sns_pain_left]
        self.somatic_map_senses_right = self.sns_odors_right[1:] + [self.sns_pain_right]
        self.somatic_map_senses = [(l + r) / 2 for l, r in
                                   zip(self.somatic_map_senses_left, self.somatic_map_senses_right)]

        # Each sensor's value relative to the total
        self.somatic_map_factors = [2 * sensor - sum(self.somatic_map_senses) for sensor in self.somatic_map_senses]

        # Replace the last factor with the computed pain value to emphasize pain
        self.somatic_map_factors[-1] = self.pain

        # Apply a sigmoid to the difference between left and right sensor values modulated by the factor
        self.somatic_map_sigmoids = [
            (r - l) / (1 + math.exp(-50 * factor))
            for l, r, factor in
            zip(self.somatic_map_senses_left, self.somatic_map_senses_right, self.somatic_map_factors)
        ]

        # Somatic map is the negative sum of the sigmoid values
        self.somatic_map = -sum(self.somatic_map_sigmoids)

        # --- Appetitive State and Turn Angle --- EXACT COPY (INCLUDING THE MISSING reward_experience!)
        self.app_state = 0.01 + (
                1 / (1 + math.exp(- (
                    1 * self.incentive - 8 * self.satiation - 0.1 * self.pain - 0.1 * self.pain_switch * self.reward_experience))) +
                0.1 * ((self.app_state_switch - 1) * 0.5)
        )

        self.app_state_switch = (-2 / (1 + math.exp(-100 * (self.app_state - 0.245)))) + 1
        self.turn_angle = 3 * (
                    (2 * self.app_state_switch) / (1 + math.exp(3 * self.somatic_map)) - self.app_state_switch)

        # --- Encounter Timer --- EXACT COPY
        if self.encounter_timer > 0:
            self.encounter_timer -= 1

        if DEBUG_MODE and encounter != "none" and self.encounter_timer == (ENCOUNTER_COOLDOWN - 1):
            print(
                "Tick encountered", encounter,
                "Flab:", self.flab_counter,
                "Hermi:", self.hermi_counter,
                "Drug:", self.drug_counter
            )
            print(
                "Satiation:", round(self.satiation, 4),
                "Nutrition:", round(self.nutrition, 2),
                "Incentive:", round(self.incentive, 2),
                "AppState:", round(self.app_state, 3),
                "AppStateSwitch:", round(self.app_state_switch, 3),
                "Somatic_Map:", round(self.somatic_map, 3),
                "Sns_Bet:", round(sns_betaine, 2),
                "Sns_Hermi:", round(sns_hermi, 2),
                "Sns_Flab:", round(sns_flab, 2),
                "Sns_Drug:", round(sns_drug, 2),
                "Vh:", round(self.Vh, 2),
                "Vf:", round(self.Vf, 2),
                "Vd:", round(self.Vd, 2)
            )

        return self.turn_angle

    def move_cyberslug(self, turn_angle):
        """Move the Cyberslug based on turn angle"""
        # Update angle
        self.angle -= 2 * turn_angle

        # Calculate new position
        new_x = self.x + self.speed * math.cos(math.radians(self.angle))
        new_y = self.y + self.speed * math.sin(math.radians(self.angle))

        # Use wrap_around function
        self.x, self.y = wrap_around(new_x, new_y)

        # Update Mesa position
        new_pos = (int(self.x), int(self.y))
        self.model.grid.move_agent(self, new_pos)

        # Update path
        self.path.append((self.x, self.y))
        if len(self.path) > 1000:
            self.path.pop(0)


class PreyAgent(Agent):
    """EXACT translation of original Prey class"""

    def __init__(self, unique_id, model, x, y, color, odorlist):
        super().__init__(unique_id, model)

        # EXACT COPY from original Prey __init__
        self.x = x
        self.y = y
        self.color = color
        self.odorlist = odorlist
        self.angle = random.uniform(0, 360)
        self.radius = PREY_RADIUS

        # Determine prey type from odorlist
        if odorlist == HERMI_ODOR:
            self.prey_type = "hermi"
        elif odorlist == FLAB_ODOR:
            self.prey_type = "flab"
        elif odorlist == DRUG_ODOR:
            self.prey_type = "drug"
        else:
            self.prey_type = "unknown"

        # Set Mesa position
        self.pos = (int(x), int(y))

    def step(self):
        """Mesa step method - calls original move"""
        self.move()
        self.deposit_odor()

    def move(self):
        """EXACT COPY of original Prey move method"""
        self.angle += random.uniform(-1, 1)
        step = 0.1
        self.x += step * math.cos(math.radians(self.angle))
        self.y += step * math.sin(math.radians(self.angle))
        self.x, self.y = wrap_around(self.x, self.y)

        # Update Mesa position
        new_pos = (int(self.x), int(self.y))
        self.model.grid.move_agent(self, new_pos)

    def deposit_odor(self):
        """Deposit odor in patch system"""
        self.model.set_patch(self.x, self.y, self.odorlist)

    def respawn(self):
        """EXACT COPY of original Prey respawn method"""
        self.x = random.randint(0, GRID_WIDTH)
        self.y = random.randint(0, GRID_HEIGHT)
        self.angle = random.uniform(0, 360)

        # Update Mesa position
        new_pos = (int(self.x), int(self.y))
        self.model.grid.move_agent(self, new_pos)


# Specific prey types - EXACT as original
class HermiAgent(PreyAgent):
    def __init__(self, unique_id, model, x, y):
        super().__init__(unique_id, model, x, y, CYAN, HERMI_ODOR)


class FlabAgent(PreyAgent):
    def __init__(self, unique_id, model, x, y):
        super().__init__(unique_id, model, x, y, PINK, FLAB_ODOR)


class FauxFlabAgent(PreyAgent):
    def __init__(self, unique_id, model, x, y):
        super().__init__(unique_id, model, x, y, YELLOW, DRUG_ODOR)
