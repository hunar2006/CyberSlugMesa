import mesa
import numpy as np
import math
import random
from config import *


class CyberslugAgent(mesa.Agent):
    """
    Biologically-accurate Cyberslug agent implementing neural decision-making
    """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        # Physical properties
        self.speed = 1
        self.heading = 0  # degrees
        self.path = [self.pos] if self.pos else []

        # Neural/Biological state
        self.nutrition = 0.5
        self.incentive = 0.0
        self.satiation = 0.0
        self.app_state = 0.0
        self.app_state_switch = 0.0
        self.somatic_map = 0.0

        # Learning variables (Value functions)
        self.Vh = 0.0  # Hermi value
        self.Vf = 0.0  # Flab value
        self.Vd = 0.0  # Drug value

        # Learning parameters
        self.alpha_hermi = ALPHA_HERMI
        self.beta_hermi = BETA_HERMI
        self.lambda_hermi = LAMBDA_HERMI
        self.alpha_flab = ALPHA_FLAB
        self.beta_flab = BETA_FLAB
        self.lambda_flab = LAMBDA_FLAB
        self.alpha_drug = ALPHA_DRUG
        self.beta_drug = BETA_DRUG
        self.lambda_drug = LAMBDA_DRUG

        # Sensory system
        self.sns_odors_left = [0.0] * 4
        self.sns_odors_right = [0.0] * 4
        self.sns_odors = [0.0] * 4

        # Pain and reward
        self.sns_pain_left = 0.0
        self.sns_pain_right = 0.0
        self.spontaneous_pain = 2.0
        self.reward_experience = 0.0

        # Encounter tracking
        self.encounter_timer = 0
        self.hermi_counter = 0
        self.flab_counter = 0
        self.drug_counter = 0

    def step(self):
        """Main behavioral step - sensing, learning, movement"""
        self.sense_environment()
        self.process_encounters()
        self.update_neural_state()
        self.move()
        self.update_timers()

    def sense_environment(self):
        """Get sensory input from environment"""
        # Get nearby odor concentrations
        sensors_left, sensors_right = self.get_sensor_readings()

        # Convert to logarithmic scale (biological sensing)
        self.sns_odors_left = [0 if i <= 1e-7 else (7 + math.log10(i)) for i in sensors_left]
        self.sns_odors_right = [0 if i <= 1e-7 else (7 + math.log10(i)) for i in sensors_right]
        self.sns_odors = [(l + r) / 2 for l, r in zip(self.sns_odors_left, self.sns_odors_right)]

    def get_sensor_readings(self):
        """Calculate sensor positions and read odor concentrations"""
        if not self.pos:
            return [0.0] * 4, [0.0] * 4

        x, y = self.pos

        # Calculate sensor positions (45 degrees offset)
        left_x = x + SENSOR_DISTANCE * math.cos(math.radians(self.heading + 45))
        left_y = y + SENSOR_DISTANCE * math.sin(math.radians(self.heading + 45))
        right_x = x + SENSOR_DISTANCE * math.cos(math.radians(self.heading - 45))
        right_y = y + SENSOR_DISTANCE * math.sin(math.radians(self.heading - 45))

        # Get odor concentrations at sensor positions
        left_odors = self.model.get_odor_at_position(left_x, left_y)
        right_odors = self.model.get_odor_at_position(right_x, right_y)

        return left_odors, right_odors

    def process_encounters(self):
        """Handle prey encounters and learning"""
        if self.encounter_timer > 0:
            return

        # Check for prey in same cell
        cellmates = self.model.grid.get_cell_list_contents([self.pos])

        for agent in cellmates:
            if isinstance(agent, PreyAgent):
                encounter_type = agent.prey_type
                self.learn_from_encounter(encounter_type)
                agent.respawn()
                self.encounter_timer = ENCOUNTER_COOLDOWN
                break

    def learn_from_encounter(self, encounter_type):
        """Update value functions based on encounter"""
        if encounter_type == "hermi":
            self.Vh += self.alpha_hermi * self.beta_hermi * (self.lambda_hermi - self.Vh)
            self.nutrition += 0.1
            self.hermi_counter += 1

        elif encounter_type == "flab":
            self.Vf += self.alpha_flab * self.beta_flab * (self.lambda_flab - self.Vf)
            self.nutrition += 0.1
            self.flab_counter += 1

        elif encounter_type == "drug":
            self.Vd += self.alpha_drug * self.beta_drug * (self.lambda_drug - self.Vd)
            self.drug_counter += 1

    def update_neural_state(self):
        """Core neural computation - the heart of Cyberslug behavior"""
        sns_betaine, sns_hermi, sns_flab, sns_drug = self.sns_odors

        # Nutrition decay
        self.nutrition -= 0.005 * self.nutrition

        # Satiation calculation
        self.satiation = 1 / ((1 + 0.7 * math.exp(-4 * self.nutrition + 2)) ** 2)

        # Pain processing
        self.sns_pain = (self.sns_pain_left + self.sns_pain_right) / 2
        self.pain = 10 / (1 + math.exp(-2 * (self.sns_pain + self.spontaneous_pain) + 10))
        self.pain_switch = 1 - 2 / (1 + math.exp(-10 * (self.sns_pain - 0.2)))

        # Reward calculations
        self.reward_pos = (
                sns_betaine / (1 + (0.05 * self.Vh * sns_hermi) - 0.006 / self.satiation) +
                3.0 * self.Vh * sns_hermi +
                8.0 * self.Vd * sns_drug
        )
        self.reward_neg = 0.59 * self.Vf * sns_flab
        self.incentive = self.reward_pos - self.reward_neg

        # Somatic map computation
        self.compute_somatic_map()

        # Appetitive state
        self.app_state = 0.01 + (
                1 / (1 + math.exp(-(1 * self.incentive - 8 * self.satiation - 0.1 * self.pain -
                                    0.1 * self.pain_switch * self.reward_experience))) +
                0.1 * ((self.app_state_switch - 1) * 0.5)
        )

        self.app_state_switch = (-2 / (1 + math.exp(-100 * (self.app_state - 0.245)))) + 1

        # Calculate turn angle
        self.turn_angle = 3 * (
                    (2 * self.app_state_switch) / (1 + math.exp(3 * self.somatic_map)) - self.app_state_switch)

    def compute_somatic_map(self):
        """Spatial decision-making computation"""
        # Sensory inputs for somatic mapping
        somatic_left = self.sns_odors_left[1:] + [self.sns_pain_left]
        somatic_right = self.sns_odors_right[1:] + [self.sns_pain_right]
        somatic_avg = [(l + r) / 2 for l, r in zip(somatic_left, somatic_right)]

        # Compute factors
        factors = [2 * sensor - sum(somatic_avg) for sensor in somatic_avg]
        factors[-1] = self.pain  # Emphasize pain

        # Sigmoid processing
        sigmoids = [
            (r - l) / (1 + math.exp(-50 * factor))
            for l, r, factor in zip(somatic_left, somatic_right, factors)
        ]

        self.somatic_map = -sum(sigmoids)

    def move(self):
        """Move based on neural computations"""
        if not self.pos:
            return

        # Update heading based on turn angle
        self.heading -= 2 * self.turn_angle

        # Calculate new position
        new_x = self.pos[0] + self.speed * math.cos(math.radians(self.heading))
        new_y = self.pos[1] + self.speed * math.sin(math.radians(self.heading))

        # Move to new position
        new_pos = self.model.grid.torus_adj((int(new_x), int(new_y)))
        self.model.grid.move_agent(self, new_pos)

        # Update path
        self.path.append(new_pos)
        if len(self.path) > 100:  # Limit path length
            self.path.pop(0)

    def update_timers(self):
        """Update internal timers"""
        if self.encounter_timer > 0:
            self.encounter_timer -= 1


class PreyAgent(mesa.Agent):
    """Base class for prey agents"""

    def __init__(self, unique_id, model, prey_type, odor_signature):
        super().__init__(unique_id, model)
        self.prey_type = prey_type
        self.odor_signature = odor_signature
        self.movement_angle = random.uniform(0, 360)

    def step(self):
        """Random movement and odor deposition"""
        self.random_move()
        self.deposit_odor()

    def random_move(self):
        """Random walk behavior"""
        self.movement_angle += random.uniform(-10, 10)

        new_x = self.pos[0] + 0.5 * math.cos(math.radians(self.movement_angle))
        new_y = self.pos[1] + 0.5 * math.sin(math.radians(self.movement_angle))

        new_pos = self.model.grid.torus_adj((int(new_x), int(new_y)))
        self.model.grid.move_agent(self, new_pos)

    def deposit_odor(self):
        """Deposit odor signature at current position"""
        self.model.deposit_odor(self.pos, self.odor_signature)

    def respawn(self):
        """Respawn at random location when consumed"""
        new_pos = self.model.grid.find_empty()
        if new_pos:
            self.model.grid.move_agent(self, new_pos)


class HermiAgent(PreyAgent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model, "hermi", HERMI_ODOR)


class FlabAgent(PreyAgent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model, "flab", FLAB_ODOR)


class FauxFlabAgent(PreyAgent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model, "drug", DRUG_ODOR)
