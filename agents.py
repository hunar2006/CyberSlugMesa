"""
agents.py - Agent definitions for CyberSlug simulation
Contains CyberslugAgent and PreyAgent classes
"""
from mesa import Agent
import math
import numpy as np

class PreyAgent(Agent):
    """
    Prey agent that moves randomly and deposits odor trails.
    Types: hermi (Hermissenda), flab (Flabellina), fauxflab (Faux-Flabellina)
    """

    def __init__(self, unique_id, model, prey_type, color, odor):
        super().__init__(model)
        self.unique_id = unique_id
        self.prey_type = prey_type
        self.color = color
        self.odor = odor
        self.angle = self.random.uniform(0, 360)
        self.radius = model.prey_radius
        self.heading = self.random.uniform(0, 360)
        self.manual_heading = False
        self.speed = 0.1

    def step(self):
        """Move and deposit odor"""
        # Get current position
        x, y = self.pos

        # Deposit odor at current location
        self.model.set_patch_odor(x, y, self.odor)

        # Move
        if self.manual_heading:
            new_x = x + self.speed * math.cos(math.radians(self.heading))
            new_y = y + self.speed * math.sin(math.radians(self.heading))
        else:
            self.angle += self.random.uniform(-1, 1)
            new_x = x + self.speed * math.cos(math.radians(self.angle))
            new_y = y + self.speed * math.sin(math.radians(self.angle))

        # Move agent (toroidal space handles wrapping)
        self.model.space.move_agent(self, (new_x, new_y))

    def respawn(self):
        """Respawn at random location"""
        new_x = self.random.randrange(self.model.width)
        new_y = self.random.randrange(self.model.height)
        self.angle = self.random.uniform(0, 360)
        self.model.space.move_agent(self, (new_x, new_y))


class CyberslugAgent(Agent):
    """
    The Cyberslug agent with learning, motivation, and sensory processing.
    Exhibits appetitive behavior and associative learning.
    """

    def __init__(self, unique_id, model):
        super().__init__(model)
        self.unique_id = unique_id

        # Position and movement
        self.angle = 0  # heading in degrees
        self.speed = 3
        self.path = []

        # Learning and motivation variables
        self.nutrition = 0.5
        self.incentive = 0.0
        self.satiation = 0.0
        self.app_state = 0.0
        self.app_state_switch = 0.0

        # Somatic map
        self.somatic_map = 0.0

        # Learning variables (associative weights)
        self.Vh = 0.0  # Value of Hermissenda odor
        self.Vf = 0.0  # Value of Flabellina odor
        self.Vd = 0.0  # Value of drug odor

        # Learning rates
        self.alpha_hermi = 0.5
        self.beta_hermi = 1.0
        self.lambda_hermi = 1.0

        self.alpha_flab = 0.5
        self.beta_flab = 1.0
        self.lambda_flab = 1.0

        self.alpha_drug = 0.5
        self.beta_drug = 1.0
        self.lambda_drug = 1.0

        # Sensor arrays (betaine, hermi, flab, drug)
        self.sns_odors_left = [0.0] * 4
        self.sns_odors_right = [0.0] * 4
        self.sns_odors = [0.0] * 4

        # Pain and reward
        self.sns_pain_left = 0.0
        self.sns_pain_right = 0.0
        self.spontaneous_pain = 2.0
        self.reward_experience = 0.0
        self.incentive_smooth = 0.0

        # Counters and timers
        self.encounter_timer = 0
        self.hermi_counter = 0
        self.flab_counter = 0
        self.fauxflab_counter = 0
        self.drug_counter = 0

    def step(self):
        """Execute one step of the Cyberslug's behavior"""
        # Get sensor readings
        x, y = self.pos
        sensors_left, sensors_right = self.model.get_sensors(x, y, self.angle)

        # Check for encounters with prey
        encounter = self.check_encounters()

        # Update internal state based on sensors and encounter
        turn_angle = self.update_state(sensors_left, sensors_right, encounter)

        # Update heading
        self.angle = (self.angle - 2 * turn_angle) % 360

        # Move forward
        new_x = x + self.speed * math.cos(math.radians(self.angle))
        new_y = y + self.speed * math.sin(math.radians(self.angle))

        # Update position (space handles toroidal wrapping)
        self.model.space.move_agent(self, (new_x, new_y))

        # Add to path for visualization
        self.path.append((new_x, new_y))
        if len(self.path) > 1000:  # Limit path length
            self.path.pop(0)

    def check_encounters(self):
        """Check if Cyberslug has encountered any prey"""
        x, y = self.pos
        encounter = "none"

        # Get all agents in a radius around the slug
        neighbors = self.model.space.get_neighbors(
            (x, y),
            radius=40,  # Detection radius (slug sprite is ~80px, so 40 is half)
            include_center=True
        )

        for neighbor in neighbors:
            if isinstance(neighbor, PreyAgent):
                # Calculate distance
                nx, ny = neighbor.pos
                distance = math.sqrt((x - nx)**2 + (y - ny)**2)

                # Check collision (slug radius ~40, prey radius ~4)
                if distance < (40 + neighbor.radius):
                    encounter = neighbor.prey_type
                    neighbor.respawn()
                    break  # Only one encounter per step

        return encounter

    def update_state(self, sensors_left, sensors_right, encounter):
        """Update internal state based on sensory input and encounters"""
        # Convert sensor values to log scale
        self.sns_odors_left = [
            0 if i <= 1e-7 else (7 + math.log10(i))
            for i in sensors_left
        ]
        self.sns_odors_right = [
            0 if i <= 1e-7 else (7 + math.log10(i))
            for i in sensors_right
        ]
        self.sns_odors = [
            (l + r) / 2
            for l, r in zip(self.sns_odors_left, self.sns_odors_right)
        ]

        sns_betaine, sns_hermi, sns_flab, sns_drug = self.sns_odors

        # --- Associative learning from prey encounters ---
        if encounter == "hermi":
            self.Vh += self.alpha_hermi * self.beta_hermi * (self.lambda_hermi - self.Vh)
            self.nutrition += 0.3
            if self.encounter_timer == 0:
                self.hermi_counter += 1
                self.encounter_timer = self.model.encounter_cooldown

        if encounter == "flab":
            self.Vf += self.alpha_flab * self.beta_flab * (self.lambda_flab - self.Vf)
            self.nutrition += 0.3
            if self.encounter_timer == 0:
                self.flab_counter += 1
                self.encounter_timer = self.model.encounter_cooldown

        if encounter == "fauxflab":
            # Negative learning - faux doesn't provide nutrition
            self.Vf += self.alpha_flab * self.beta_flab * (0.0 - self.Vf)
            if self.encounter_timer == 0:
                self.fauxflab_counter += 1
                self.encounter_timer = self.model.encounter_cooldown

        # --- Pain calculations ---
        self.sns_pain = (self.sns_pain_left + self.sns_pain_right) / 2
        self.pain = 10 / (1 + math.exp(-2 * (self.sns_pain + self.spontaneous_pain) + 10))
        self.pain_switch = 1 - 2 / (1 + math.exp(-10 * (self.sns_pain - 0.2)))

        # --- Nutrition, Satiation, and Incentive ---
        self.nutrition -= 0.005 * self.nutrition
        self.satiation = 1 / ((1 + 0.7 * math.exp(-4 * self.nutrition + 2)) ** 2)

        self.reward_pos = (
            sns_betaine / (1 + (0.05 * self.Vh * sns_hermi) - 0.006 / self.satiation)
            + 3.0 * self.Vh * sns_hermi
            + 8.0 * self.Vd * sns_drug
        )
        self.reward_neg = 0.59 * self.Vf * sns_flab
        self.incentive = self.reward_pos - self.reward_neg

        # --- Somatic Map Calculation ---
        self.somatic_map_senses_left = self.sns_odors_left[1:] + [self.sns_pain_left]
        self.somatic_map_senses_right = self.sns_odors_right[1:] + [self.sns_pain_right]
        self.somatic_map_senses = [
            (l + r) / 2
            for l, r in zip(self.somatic_map_senses_left, self.somatic_map_senses_right)
        ]

        # Each sensor's value relative to the total
        self.somatic_map_factors = [
            2 * sensor - sum(self.somatic_map_senses)
            for sensor in self.somatic_map_senses
        ]
        # Emphasize pain
        self.somatic_map_factors[-1] = self.pain

        # Apply sigmoid to left-right differences
        self.somatic_map_sigmoids = [
            (r - l) / (1 + math.exp(-50 * factor))
            for l, r, factor in zip(
                self.somatic_map_senses_left,
                self.somatic_map_senses_right,
                self.somatic_map_factors
            )
        ]
        # Somatic map is negative sum
        self.somatic_map = -sum(self.somatic_map_sigmoids)

        # --- Appetitive State and Turn Angle ---
        self.app_state = 0.01 + (
            1 / (1 + math.exp(-(
                1 * self.incentive
                - 8 * self.satiation
                - 0.1 * self.pain
                - 0.1 * self.pain_switch * self.reward_experience
            )))
            + 0.1 * ((self.app_state_switch - 1) * 0.5)
        )

        self.app_state_switch = (-2 / (1 + math.exp(-100 * (self.app_state - 0.245)))) + 1
        turn_angle = 3 * (
            (2 * self.app_state_switch) / (1 + math.exp(3 * self.somatic_map))
            - self.app_state_switch
        )

        # --- Encounter Timer ---
        if self.encounter_timer > 0:
            self.encounter_timer -= 1

        return turn_angle