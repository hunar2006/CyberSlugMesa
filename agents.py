"""
agents.py - Agent definitions for CyberSlug simulation with social interactions
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
    The Cyberslug agent with learning, motivation, sensory processing, and SOCIAL BEHAVIORS.
    Exhibits appetitive behavior, associative learning, and can bite other slugs.
    """

    def __init__(self, unique_id, model):
        super().__init__(model)
        self.unique_id = unique_id

        # Position and movement
        self.angle = 0  # heading in degrees
        self.speed = 3
        self.path = []
        self.size = 5 + self.random.uniform(0, 10)  # Variable size (5-15)

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

        # Sensor arrays (betaine, hermi, flab, drug, SLUG)
        self.sns_odors_left = [0.0] * 5  # Added 5th for slug odor
        self.sns_odors_right = [0.0] * 5
        self.sns_odors = [0.0] * 5

        # Pain and reward
        self.sns_pain_left = 0.0
        self.sns_pain_right = 0.0
        self.sns_pain_total = 0.0
        self.spontaneous_pain = 2.0
        self.reward_experience = 0.0
        self.incentive_smooth = 0.0

        # Counters and timers
        self.encounter_timer = 0
        self.hermi_counter = 0
        self.flab_counter = 0
        self.fauxflab_counter = 0
        self.drug_counter = 0

        # SOCIAL BEHAVIORS - New attributes
        self.bite_counter = 0
        self.被咬_counter = 0  # Times been bitten
        self.is_biting = False
        self.bite_target = None
        self.bite_cooldown = 0
        self.pain_from_bite = 0.0

        # Habituation to conspecific odor
        self.M_conspecific = 0  # Processed conspecific odor (habituated)
        self.W3_conspecific = 0.5  # Synaptic weight for habituation

    def step(self):
        """Execute one step of the Cyberslug's behavior"""
        # Get sensor readings
        x, y = self.pos
        sensors_left, sensors_right = self.model.get_sensors(x, y, self.angle)

        # Check for encounters with prey
        encounter = self.check_encounters()

        # Check for social interactions with other slugs
        self.check_slug_interactions()

        # Update internal state based on sensors and encounter
        turn_angle = self.update_state(sensors_left, sensors_right, encounter)

        # Update heading
        self.angle = (self.angle - 2 * turn_angle) % 360

        # Move forward (speed affected by pain)
        base_speed = self.speed
        pain_speed_boost = self.pain_from_bite / 20  # Pain makes you move faster
        actual_speed = base_speed + pain_speed_boost

        new_x = x + actual_speed * math.cos(math.radians(self.angle))
        new_y = y + actual_speed * math.sin(math.radians(self.angle))

        # Update position (space handles toroidal wrapping)
        self.model.space.move_agent(self, (new_x, new_y))

        # Add to path for visualization
        self.path.append((new_x, new_y))
        if len(self.path) > 1000:  # Limit path length
            self.path.pop(0)

        # Decay pain from bites
        self.pain_from_bite *= 0.8

        # Decay bite cooldown
        if self.bite_cooldown > 0:
            self.bite_cooldown -= 1

    def check_slug_interactions(self):
        """Check for interactions with other slugs - SOCIAL BEHAVIOR"""
        x, y = self.pos
        self.is_biting = False

        # Get nearby slugs
        neighbors = self.model.space.get_neighbors(
            (x, y),
            radius=50,  # Detection radius for other slugs
            include_center=False
        )

        for neighbor in neighbors:
            if isinstance(neighbor, CyberslugAgent) and neighbor != self:
                nx, ny = neighbor.pos
                distance = math.sqrt((x - nx)**2 + (y - ny)**2)

                # Check if close enough to bite (within cone in front)
                # Calculate if neighbor is in front cone
                angle_to_neighbor = math.degrees(math.atan2(ny - y, nx - x))
                angle_diff = abs((angle_to_neighbor - self.angle + 180) % 360 - 180)

                if distance < (0.4 * self.size + 0.4 * neighbor.size) and angle_diff < 45:
                    # Close enough and in front - attempt to bite!
                    if self.should_bite(neighbor):
                        self.bite_slug(neighbor)
                        break

    def should_bite(self, other_slug):
        """Determine if this slug should bite another - based on size, arousal, and cooldown"""
        if self.bite_cooldown > 0:
            return False

        # Larger slugs are more likely to bite
        size_advantage = self.size > other_slug.size

        # Higher appetitive state makes more aggressive
        aroused = self.app_state > 0.5

        # High processed conspecific odor + satiation = guarding behavior
        guarding = (self.M_conspecific > 0.01 * self.size / 4) and (self.satiation > 0.7)

        return (size_advantage and aroused) or guarding

    def bite_slug(self, target):
        """Bite another slug - inflict pain"""
        self.is_biting = True
        self.bite_target = target
        self.bite_counter += 1
        self.bite_cooldown = 10  # Can't bite again for 10 steps

        # Calculate pain inflicted based on size
        pain_amount = 20.0 * (self.size / 10)

        # Inflict pain on target
        target.pain_from_bite += pain_amount
        target.被咬_counter += 1

    def check_encounters(self):
        """Check if Cyberslug has encountered any prey"""
        x, y = self.pos
        encounter = "none"

        # Get all agents in a radius around the slug
        neighbors = self.model.space.get_neighbors(
            (x, y),
            radius=40,  # Detection radius
            include_center=True
        )

        for neighbor in neighbors:
            if isinstance(neighbor, PreyAgent):
                # Calculate distance
                nx, ny = neighbor.pos
                distance = math.sqrt((x - nx)**2 + (y - ny)**2)

                # Check collision
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

        sns_betaine, sns_hermi, sns_flab, sns_drug, sns_slug = self.sns_odors

        # Habituation to conspecific (slug) odor
        self.habituate_to_conspecific(sns_slug)

        # --- Associative learning from prey encounters ---
        if encounter == "hermi":
            self.Vh += self.alpha_hermi * self.beta_hermi * (self.lambda_hermi - self.Vh)
            self.nutrition += 0.3
            self.size = min(self.size + 0.1, self.model.max_slug_size)  # Grow slightly
            if self.encounter_timer == 0:
                self.hermi_counter += 1
                self.encounter_timer = self.model.encounter_cooldown

        if encounter == "flab":
            self.Vf += self.alpha_flab * self.beta_flab * (self.lambda_flab - self.Vf)
            self.nutrition += 0.3
            self.size = min(self.size + 0.1, self.model.max_slug_size)
            if self.encounter_timer == 0:
                self.flab_counter += 1
                self.encounter_timer = self.model.encounter_cooldown

        if encounter == "fauxflab":
            # Negative learning - faux doesn't provide nutrition
            self.Vf += self.alpha_flab * self.beta_flab * (0.0 - self.Vf)
            if self.encounter_timer == 0:
                self.fauxflab_counter += 1
                self.encounter_timer = self.model.encounter_cooldown

        # --- Pain calculations (includes pain from bites) ---
        self.sns_pain_total = (self.sns_pain_left + self.sns_pain_right) / 2 + self.pain_from_bite
        self.pain = 10 / (1 + math.exp(-2 * (self.sns_pain_total + self.spontaneous_pain) + 10))
        self.pain_switch = 1 - 2 / (1 + math.exp(-10 * (self.sns_pain_total - 0.2)))

        # --- Nutrition, Satiation, and Incentive ---
        self.nutrition -= 0.005 * self.nutrition
        self.satiation = 1 / ((1 + 0.7 * math.exp(-4 * self.nutrition + 2)) ** 2)

        self.reward_pos = (
            sns_betaine / (1 + (0.05 * self.Vh * sns_hermi) - 0.006 / self.satiation)
            + 3.0 * self.Vh * sns_hermi
            + 8.0 * self.Vd * sns_drug
        )
        self.reward_neg = 0.59 * self.Vf * sns_flab + self.sns_pain_total
        self.incentive = self.reward_pos - self.reward_neg

        # --- Somatic Map Calculation (includes conspecific odor) ---
        # Add processed conspecific odor to somatic map calculation
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
        # Include processed conspecific odor in appetitive state
        # At high satiation + high conspecific odor = guarding behavior
        self.app_state = 0.01 + (
            1 / (1 + math.exp(-(
                1 * self.incentive
                - 8 * self.satiation
                - 0.1 * self.pain
                - 0.1 * self.pain_switch * self.reward_experience
                + 0.9 * self.M_conspecific  # Conspecific odor increases arousal
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

    def habituate_to_conspecific(self, sns_slug):
        """Habituation/sensitization to conspecific (slug) odor - like in NetLogo"""
        # Based on NetLogo's calc-SH function
        S_conspecific = sns_slug
        M0_conspecific = self.size / 4  # Baseline activity

        # Simple habituation: processed odor decreases with repeated exposure
        self.M_conspecific = self.W3_conspecific * S_conspecific

        # Update synaptic weight (habituation)
        dW3 = ((M0_conspecific / (S_conspecific + 0.01)) - (self.M_conspecific / (S_conspecific + 0.01))) / 10
        self.W3_conspecific = max(0.1, min(1.0, self.W3_conspecific + dW3))