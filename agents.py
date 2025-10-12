"""
agents.py - Agent definitions for CyberSlug simulation
Enhanced with ALL NetLogo features:
- Advanced learning circuit (R+, R-, NR neurons)
- Proboscis behavior
- Sophisticated sensor system (7 nociceptors)
- Drug odor support
- Collision detection
- Eligibility traces
- Dynamic baselines
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
        self.speed = 0.02  # NetLogo speed

        # Clustering support
        self.cluster_target = None

    def step(self):
        """Move and deposit odor"""
        x, y = self.pos

        # Deposit odor at current location
        self.model.set_patch_odor(x, y, self.odor)

        # Move with clustering behavior if enabled
        if self.model.clustering and self.cluster_target:
            self.move_to_cluster()
        else:
            # Random movement
            if self.manual_heading:
                new_x = x + self.speed * math.cos(math.radians(self.heading))
                new_y = y + self.speed * math.sin(math.radians(self.heading))
            else:
                self.angle += self.random.uniform(-1, 1)
                new_x = x + self.speed * math.cos(math.radians(self.angle))
                new_y = y + self.speed * math.sin(math.radians(self.angle))

            # Move agent (toroidal space handles wrapping)
            if not self.model.immobilize:
                self.model.space.move_agent(self, (new_x, new_y))

    def move_to_cluster(self):
        """Move towards cluster center with some randomness"""
        x, y = self.pos
        cx, cy = self.cluster_target

        # Calculate distance to cluster
        dx = cx - x
        dy = cy - y
        dist = math.sqrt(dx**2 + dy**2)

        # If outside cluster radius, move towards center
        if dist > self.model.cluster_radius:
            angle_to_cluster = math.degrees(math.atan2(dy, dx))
            # Gradually turn towards cluster
            angle_diff = (angle_to_cluster - self.angle + 180) % 360 - 180
            self.angle += angle_diff / 5
        else:
            # Inside cluster, move randomly with sine wave
            self.angle += 2 * math.sin(math.radians(30 * self.model.ticks)) - 4 + self.random.uniform(-8, 8)

        # Move forward
        speed = 0.05
        new_x = x + speed * math.cos(math.radians(self.angle))
        new_y = y + speed * math.sin(math.radians(self.angle))

        if not self.model.immobilize:
            self.model.space.move_agent(self, (new_x, new_y))

    def respawn(self):
        """Respawn at cluster location if clustering, otherwise random"""
        if self.model.clustering and self.cluster_target:
            cx, cy = self.cluster_target
            new_x = cx + self.random.uniform(-self.model.cluster_radius, self.model.cluster_radius)
            new_y = cy + self.random.uniform(-self.model.cluster_radius, self.model.cluster_radius)
        else:
            new_x = self.random.randrange(self.model.width)
            new_y = self.random.randrange(self.model.height)

        self.angle = self.random.uniform(0, 360)
        self.model.space.move_agent(self, (new_x, new_y))


class Nociceptor:
    """Pain receptor with position and pain value"""
    def __init__(self, id_name, parent):
        self.id = id_name
        self.parent = parent
        self.x = 0
        self.y = 0
        self.painval = 0.0
        self.hit = False


class CyberslugAgent(Agent):
    """
    The Cyberslug agent with COMPLETE NetLogo implementation:
    - Advanced learning circuit (R+, R-, NR neurons)
    - Proboscis extension
    - 7 nociceptor pain sensors
    - Eligibility traces
    - Dynamic baselines
    - Collision detection
    - Social behaviors
    """

    def __init__(self, unique_id, model):
        super().__init__(model)
        self.unique_id = unique_id

        # Position and movement
        self.angle = 0  # heading in degrees
        self.previous_heading = 0
        self.speed = 0.06
        self.path = []
        self.size = 5 + self.random.uniform(0, 10)  # Variable size (5-15)
        self.tick_timer = 10

        # Learning and motivation variables
        self.nutrition = 0.5
        self.incentive = 0.0
        self.satiation = 0.5
        self.app_state = 0.0
        self.app_state_switch = 0.0

        # Somatic map
        self.somatic_map = 0.0

        # ADVANCED LEARNING CIRCUIT - NetLogo style
        # Association strengths (V) for each pathway
        self.Vh_rp = 0.0  # Hermi -> R+ (positive reward)
        self.Vh_rn = 0.0  # Hermi -> R- (negative reward)
        self.Vh_n = 0.0   # Hermi -> NR (non-reward)
        self.Vf_rp = 0.0  # Flab -> R+
        self.Vf_rn = 0.0  # Flab -> R-
        self.Vf_n = 0.0   # Flab -> NR

        # Baselines for association strengths (V0)
        self.Vh_rp0 = 0.0
        self.Vh_rn0 = 0.0
        self.Vh_n0 = 0.0
        self.Vf_rp0 = 0.0
        self.Vf_rn0 = 0.0
        self.Vf_n0 = 0.0

        # Synaptic weights (W) calculated from V
        self.Wh_rp = 0.0
        self.Wh_rn = 0.0
        self.Wh_n = 0.0
        self.Wf_rp = 0.0
        self.Wf_rn = 0.0
        self.Wf_n = 0.0

        # Saturation flags
        self.Wh_rp_saturated = 0
        self.Wh_rn_saturated = 0
        self.Wh_n_saturated = 0
        self.Wf_rp_saturated = 0
        self.Wf_rn_saturated = 0
        self.Wf_n_saturated = 0

        # Reward neurons
        self.CS1 = 0.0  # Hermi conditioned stimulus
        self.CS2 = 0.0  # Flab conditioned stimulus
        self.R_pos = 0.0  # Positive reward neuron
        self.R_neg = 0.0  # Negative reward neuron
        self.NR = 0.0  # Non-reward neuron
        self.NR_spontaneous = 1.0

        # Reward inputs (for eligibility traces)
        self.R_pos_input = 0.0
        self.R_neg_input = 0.0

        # Sensor arrays (betaine, hermi, flab, drug, SLUG)
        self.sns_odors_left = [0.0] * 5
        self.sns_odors_right = [0.0] * 5
        self.sns_odors = [0.0] * 5

        # Sensor weights (NetLogo: oral veil is 2x more sensitive)
        self.OV_weight = 2.0  # Oral veil sensors
        self.PB_weight = 1.0  # Body sensors

        # Pain - 7 nociceptors (NetLogo style)
        self.nociceptors = []
        nociceptor_ids = ["snsrOL", "snsrOR", "snsrUL", "snsrUR", "snsrBL", "snsrBR", "snsrBM"]
        for noc_id in nociceptor_ids:
            self.nociceptors.append(Nociceptor(noc_id, self))

        self.sns_pain_left = 0.0
        self.sns_pain_right = 0.0
        self.sns_pain_caud = 0.0  # Caudal (back) pain
        self.sns_pain_total = 0.0
        self.pain = 0.0
        self.pain_switch = 0.0
        self.spontaneous_pain = 2.0

        # Reward values
        self.reward = 0.0
        self.reward_neg = 0.0
        self.reward_experience = 0.0
        self.incentive_smooth = 0.0

        # Counters and timers
        self.encounter_timer = 0
        self.hermi_counter = 0
        self.flab_counter = 0
        self.fauxflab_counter = 0
        self.drug_counter = 0

        # SOCIAL BEHAVIORS
        self.bite_counter = 0
        self.被咬_counter = 0  # Times been bitten
        self.is_biting = False
        self.bite_target = None
        self.bite_cooldown = 0
        self.pain_from_bite = 0.0
        self.collision = 0  # Collision detection

        # Habituation to conspecific odor (NetLogo circuit)
        self.Is = 0.0  # Sensory input (pleuro odor)
        self.Im = 0.0  # Modulatory input (pain)
        self.S = 0.0
        self.IN = 0.0
        self.M = 0.0   # Processed conspecific odor
        self.M0 = self.size / 4  # Baseline activity
        self.W1 = 1.0
        self.W2 = 1.0
        self.W3 = 0.5
        self.dW3 = 0.0

        # PROBOSCIS
        self.proboscis_phase = 0  # Extension phase (0-19)
        self.proboscis_extended = False

    def step(self):
        """Execute one step of the Cyberslug's behavior"""
        # Update tick timer for visual updates
        self.tick_timer -= 1
        if self.tick_timer == 0:
            self.previous_heading = self.angle
            self.tick_timer = 10

        # Get sensor readings
        x, y = self.pos
        self.update_sensors()

        # Update nociceptor positions
        self.update_nociceptor_positions()

        # Check for encounters with prey
        encounter = self.check_encounters()

        # Check for social interactions with other slugs
        self.check_slug_interactions()

        # Update internal state based on sensors and encounter
        turn_angle = self.update_state(encounter)

        # Update heading
        self.angle = (self.angle + turn_angle) % 360

        # Move forward (speed affected by pain and satiation)
        base_speed = 0.10
        pain_speed_boost = self.sns_pain_caud / 20  # Caudal pain makes you move faster
        satiation_slowdown = self.satiation / 20  # High satiation slows you down
        actual_speed = base_speed + pain_speed_boost - satiation_slowdown

        new_x = x + actual_speed * math.cos(math.radians(self.angle))
        new_y = y + actual_speed * math.sin(math.radians(self.angle))

        # Update position (space handles toroidal wrapping)
        if not self.model.immobilize:
            self.model.space.move_agent(self, (new_x, new_y))

        # Add to path for visualization
        self.path.append((new_x, new_y))
        if len(self.path) > 1000:  # Limit path length
            self.path.pop(0)

        # Decay pain from bites and external sources
        for noc in self.nociceptors:
            noc.painval *= 0.20  # NetLogo: 0.20 decay
        self.pain_from_bite *= 0.8

        # Decay bite cooldown
        if self.bite_cooldown > 0:
            self.bite_cooldown -= 1

        # Update proboscis
        self.update_proboscis()

    def update_nociceptor_positions(self):
        """Update positions of all 7 nociceptors (NetLogo style)"""
        x, y = self.pos
        heading = self.angle
        size = self.size

        positions = {
            "snsrOL": (40, 0.4),   # Oral left
            "snsrOR": (-40, 0.4),  # Oral right (negative angle = right)
            "snsrUL": (100, 0.3),  # Upper left
            "snsrUR": (-100, 0.3), # Upper right
            "snsrBL": (150, 0.35), # Back left
            "snsrBR": (-150, 0.35),# Back right
            "snsrBM": (180, 0.46)  # Back middle
        }

        for noc in self.nociceptors:
            angle_offset, distance_mult = positions[noc.id]
            # NetLogo uses: x + dist * sin(heading + offset)
            # Convert to our coordinate system
            sensor_angle = heading + angle_offset
            noc.x = x + (distance_mult * size) * math.cos(math.radians(sensor_angle))
            noc.y = y + (distance_mult * size) * math.sin(math.radians(sensor_angle))

    def update_sensors(self):
        """Update sensory input from odor patches (NetLogo style with OV/PB weights)"""
        x, y = self.pos
        heading = self.angle
        size = self.size

        # Get odor readings at each sensor location
        def get_sensor_odors(angle_offset, distance_mult):
            sensor_angle = heading + angle_offset
            sx = x + (distance_mult * size) * math.cos(math.radians(sensor_angle))
            sy = y + (distance_mult * size) * math.sin(math.radians(sensor_angle))
            return self.model.get_odor_at_position(sx, sy)

        # Get odors at each sensor (with appropriate weights)
        # Oral veil sensors (OV_weight = 2)
        odor_OL = get_sensor_odors(40, 0.4)
        odor_OR = get_sensor_odors(-40, 0.4)

        # Upper body sensors (PB_weight = 1)
        odor_UL = get_sensor_odors(100, 0.3)
        odor_UR = get_sensor_odors(-100, 0.3)

        # Back sensors (PB_weight = 1)
        odor_BL = get_sensor_odors(150, 0.35)
        odor_BR = get_sensor_odors(-150, 0.35)

        # Calculate left/right odor sensations with weights
        # odor format: [betaine, hermi, flab, drug, pleur]
        def weighted_sum(sensors_list, weights_list):
            result = [0.0] * 5
            for sensor, weight in zip(sensors_list, weights_list):
                for i in range(5):
                    result[i] += weight * sensor[i]
            return result

        # Left side: OL (oral, weight=2), UL, BL (upper/back, weight=1 each)
        odor_left = weighted_sum([odor_OL, odor_UL, odor_BL],
                                  [self.OV_weight, self.PB_weight, self.PB_weight])

        # Right side: OR (oral, weight=2), UR, BR (upper/back, weight=1 each)
        odor_right = weighted_sum([odor_OR, odor_UR, odor_BR],
                                   [self.OV_weight, self.PB_weight, self.PB_weight])

        # Convert to log scale (NetLogo style)
        def to_log_scale(odors):
            return [0 if val <= 1e-7 else (7 + math.log10(val)) for val in odors]

        self.sns_odors_left = to_log_scale(odor_left)
        self.sns_odors_right = to_log_scale(odor_right)
        self.sns_odors = [(l + r) / 2 for l, r in zip(self.sns_odors_left, self.sns_odors_right)]

        # Update pain sensors
        self.update_pain_sensors()

    def update_pain_sensors(self):
        """Calculate pain sensation from nociceptors"""
        # Pain on left side (OL, UL, BL)
        left_nocs = [n for n in self.nociceptors if n.id in ["snsrOL", "snsrUL", "snsrBL"]]
        self.sns_pain_left = sum(n.painval for n in left_nocs)

        # Pain on right side (OR, UR, BR)
        right_nocs = [n for n in self.nociceptors if n.id in ["snsrOR", "snsrUR", "snsrBR"]]
        self.sns_pain_right = sum(n.painval for n in right_nocs)

        # Pain at caudal end (BL, BR, BM)
        caud_nocs = [n for n in self.nociceptors if n.id in ["snsrBL", "snsrBR", "snsrBM"]]
        self.sns_pain_caud = sum(n.painval for n in caud_nocs)

        # Total pain
        self.sns_pain_total = (self.sns_pain_left + self.sns_pain_right) / 2

    def update_proboscis(self):
        """Update proboscis extension (NetLogo style)"""
        sns_betaine_left = self.sns_odors_left[0]
        sns_betaine_right = self.sns_odors_right[0]

        # Extend if high betaine OR high conspecific odor
        if (sns_betaine_left > 5.5 or sns_betaine_right > 5.5 or
            self.M > 1.3 * self.M0):
            self.proboscis_phase = (self.proboscis_phase + 1) % 20
            self.proboscis_extended = True
        else:
            self.proboscis_phase = 0
            self.proboscis_extended = False

    def check_slug_interactions(self):
        """Check for interactions with other slugs - SOCIAL BEHAVIOR with collision"""
        x, y = self.pos
        self.is_biting = False
        self.collision = 0

        # Get nearby slugs
        neighbors = self.model.space.get_neighbors(
            (x, y),
            radius=50,
            include_center=False
        )

        for neighbor in neighbors:
            if isinstance(neighbor, CyberslugAgent) and neighbor != self:
                nx, ny = neighbor.pos
                distance = math.sqrt((x - nx)**2 + (y - ny)**2)

                # Check if in bite cone (0.7 * size, 45 degrees)
                angle_to_neighbor = math.degrees(math.atan2(ny - y, nx - x))
                angle_diff = abs((angle_to_neighbor - self.angle + 180) % 360 - 180)

                if distance < (0.7 * self.size) and angle_diff < 45:
                    self.collision = 1  # Collision detected

                    # Bite if M > M0 (NetLogo condition: high conspecific odor)
                    if self.M > 1.01 * self.M0 and self.model.biting:
                        self.bite_slug(neighbor)
                        break

    def bite_slug(self, target):
        """Bite another slug - inflict pain at their nociceptors"""
        self.is_biting = True
        self.bite_target = target
        self.bite_counter += 1
        self.bite_cooldown = 10

        # Calculate bite position (front of this slug)
        bite_x = self.pos[0] + (0.1 * self.size) * math.cos(math.radians(self.angle))
        bite_y = self.pos[1] + (0.1 * self.size) * math.sin(math.radians(self.angle))

        # Apply pain to target's nociceptors based on distance (NetLogo style)
        target.被咬_counter += 1
        for noc in target.nociceptors:
            dist = math.sqrt((noc.x - bite_x)**2 + (noc.y - bite_y)**2)
            noc.painval += 20.0 / (dist + 0.1)
            noc.hit = True

    def check_encounters(self):
        """Check if Cyberslug has encountered any prey"""
        x, y = self.pos
        encounter = "none"

        # Get all agents in a radius around the slug
        neighbors = self.model.space.get_neighbors(
            (x, y),
            radius=40,
            include_center=True
        )

        for neighbor in neighbors:
            if isinstance(neighbor, PreyAgent):
                # Calculate distance
                nx, ny = neighbor.pos
                distance = math.sqrt((x - nx)**2 + (y - ny)**2)

                # Check collision (within bite cone)
                angle_to_prey = math.degrees(math.atan2(ny - y, nx - x))
                angle_diff = abs((angle_to_prey - self.angle + 180) % 360 - 180)

                if distance < (0.4 * self.size) and angle_diff < 45:
                    encounter = neighbor.prey_type
                    neighbor.respawn()
                    break  # Only one encounter per step

        return encounter

    def update_state(self, encounter):
        """Update internal state based on sensory input and encounters (FULL NetLogo version)"""
        sns_betaine, sns_hermi, sns_flab, sns_drug, sns_pleur = self.sns_odors
        sns_betaine_left, sns_hermi_left, sns_flab_left, sns_drug_left, sns_pleur_left = self.sns_odors_left
        sns_betaine_right, sns_hermi_right, sns_flab_right, sns_drug_right, sns_pleur_right = self.sns_odors_right

        # Habituation/sensitization to conspecific (slug) odor
        self.calc_SH()

        # --- Associative learning from prey encounters (NetLogo style) ---
        if encounter == "hermi":
            self.nutrition += 0.3
            self.size = min(self.size + 0.1, self.model.max_slug_size)
            if self.encounter_timer == 0:
                self.hermi_counter += 1
                self.encounter_timer = self.model.encounter_cooldown
                # Set reward input for learning (NetLogo: R_pos_input += 2)
                self.R_pos_input += 2.0

        if encounter == "flab":
            self.nutrition += 0.3
            self.size = min(self.size + 0.1, self.model.max_slug_size)
            if self.encounter_timer == 0:
                self.flab_counter += 1
                self.encounter_timer = self.model.encounter_cooldown
                # Set negative reward input (NetLogo: R_neg_input += 2)
                self.R_neg_input += 2.0

        if encounter == "fauxflab":
            # Faux provides nutrition but triggers learning
            self.nutrition += 0.3
            if self.encounter_timer == 0:
                self.fauxflab_counter += 1
                self.encounter_timer = self.model.encounter_cooldown
                # NetLogo: fauxflab triggers R_pos_input (REMOVE THIS in final version)
                # For now, following NetLogo exactly:
                self.R_pos_input += 2.0

        # --- Pain calculations ---
        self.pain = 10 / (1 + math.exp(-2 * (self.sns_pain_total + self.spontaneous_pain) + 10))
        self.pain_switch = 1 - 2 / (1 + math.exp(-10 * (self.sns_pain_total - 0.2)))

        # --- Nutrition, Satiation, and Incentive (NetLogo formulas) ---
        self.nutrition = self.nutrition - 0.0005 * self.nutrition

        # Satiation calculation
        if self.model.fix_satiation_override:
            self.satiation = self.model.fix_satiation_value
        else:
            self.satiation = 1 / ((1 + 0.7 * math.exp(-4 * self.nutrition + 2)) ** 2)

        # Positive reward (NetLogo formula with Vh_rp and Vf_rp)
        self.reward = (sns_betaine / (1 + (0.5 * (self.Vh_rp * sns_hermi) +
                       0 * (self.Vf_rp * sns_flab)) - 0.008 / self.satiation) +
                       1.32 * (self.Vh_rp * sns_hermi) +
                       0 * (self.Vf_rp * sns_flab))

        # Negative reward (NetLogo formula with Vf_rn)
        self.reward_neg = 1.32 * self.Vf_rn * sns_flab + self.sns_pain_total

        # Incentive salience
        self.incentive = self.reward - self.reward_neg

        # --- Somatic Map Calculation (NetLogo style) ---
        H = sns_hermi - sns_flab - 0.03 * self.M - self.sns_pain_total
        F = sns_flab - sns_hermi - 0.03 * self.M - self.sns_pain_total
        G = self.M - sns_hermi - sns_flab - self.sns_pain_total
        P = self.sns_pain_total

        self.somatic_map = -(
            (sns_hermi_left - sns_hermi_right) / (1 + math.exp(-50 * H)) +
            (sns_flab_left - sns_flab_right) / (1 + math.exp(-50 * F)) +
            (sns_pleur_left - sns_pleur_right) / (1 + math.exp(-50 * G)) +
            (self.sns_pain_left - self.sns_pain_right) / (1 + math.exp(-50 * P))
        )

        # --- Appetitive State Switch (NetLogo formula with guarding behavior) ---
        # The sns_betaine term enables guarding behavior when satiated
        app_state_pre = 0.01 + (1 / (1 + math.exp(-(
            self.incentive * 0.6 +
            0.9 * self.M +
            10 * self.satiation * (sns_betaine - 5.4)
        ))) + 0.05 * (self.app_state_switch - 1))

        self.app_state = app_state_pre
        self.app_state_switch = (-2 / (1 + math.exp(-100 * (self.app_state - 0.245)))) + 1

        # --- Turn Angle (NetLogo formula) ---
        turn_angle = self.app_state_switch * 2 * ((1 / (1 + math.exp(3 * self.somatic_map))) - 0.5)

        # --- ADVANCED LEARNING CIRCUIT ---
        self.calc_learning_circuit()

        # --- Encounter Timer ---
        if self.encounter_timer > 0:
            self.encounter_timer -= 1

        return turn_angle

    def calc_SH(self):
        """Calculate habituation/sensitization (NetLogo circuit)"""
        # Sensory inputs
        self.Is = (self.sns_odors_left[4] + self.sns_odors_right[4]) / 2  # Pleuro odor
        self.Im = (self.sns_pain_left + self.sns_pain_right) / 2  # Pain

        # Neural activity
        self.S = self.W1 * self.Is
        self.IN = self.W2 * self.Im
        self.M = self.W3 * self.S

        # Update synaptic weight W3 (habituation)
        self.dW3 = ((self.M0 / (self.S + 0.01)) - (self.M / (self.S + 0.01)) + 10 * self.IN) / 10
        self.W3 = max(0.1, min(1.0, self.W3 + self.dW3))

    def calc_learning_circuit(self):
        """
        ADVANCED LEARNING CIRCUIT (NetLogo implementation)
        - CS neurons with eligibility traces
        - R+, R-, NR neurons
        - Association strengths (V) and synaptic weights (W)
        - Dynamic baselines (V0)
        """
        sns_hermi = self.sns_odors[1]
        sns_flab = self.sns_odors[2]

        # Set CS neuron activity based on odor sensation
        if sns_hermi > 0:
            self.CS1 = sns_hermi
        if sns_flab > 0:
            self.CS2 = sns_flab

        # Decay eligibility traces for CS neurons (NetLogo: 0.8 decay)
        self.CS1 *= 0.8
        self.CS2 *= 0.8

        # Decay reward inputs (eligibility traces)
        self.R_pos_input *= 0.8
        self.R_neg_input *= 0.8

        # Calculate reward neuron activities (NetLogo formulas with sigmoids)
        # R+ neuron: receives input from CS1, CS2, reward input, inhibited by NR
        self.R_pos = 1 / (1 + math.exp(-10 * (
            self.Wh_rp * self.CS1 +
            self.Wf_rp * self.CS2 -
            0.5 * self.NR +
            self.R_pos_input
        ) + 8))

        # R- neuron: receives input from CS2, CS1, negative reward input, inhibited by NR
        self.R_neg = 1 / (1 + math.exp(-10 * (
            self.Wf_rn * self.CS2 +
            self.Wh_rn * self.CS1 -
            0.5 * self.NR +
            self.R_neg_input
        ) + 8))

        # NR neuron: receives input from CS1, CS2, spontaneous activity, inhibited by R+ and R-
        self.NR = 1 / (1 + math.exp(-4 * (
            self.Wh_n * self.CS1 +
            self.Wf_n * self.CS2 -
            self.R_pos -
            self.R_neg +
            2 * self.NR_spontaneous
        ) + 7))

        # Calculate changes in association strength (NetLogo: learning rate 0.1)
        dVh_rp = 0.1 * self.CS1 * self.R_pos
        dVh_rn = 0.1 * self.CS1 * self.R_neg
        dVf_rp = 0.1 * self.CS2 * self.R_pos
        dVf_rn = 0.1 * self.CS2 * self.R_neg
        dVh_n = 0.1 * self.CS1 * self.NR
        dVf_n = 0.1 * self.CS2 * self.NR

        # Update association strengths (only when not saturated)
        self.Vh_rp += (1 - self.Wh_rp) * dVh_rp
        self.Vh_rn += (1 - self.Wh_rn) * dVh_rn
        self.Vf_rp += (1 - self.Wf_rp) * dVf_rp
        self.Vf_rn += (1 - self.Wf_rn) * dVf_rn
        self.Vh_n += (1 - self.Wh_n) * dVh_n
        self.Vf_n += (1 - self.Wf_n) * dVf_n

        # Apply forgetting (NetLogo: constant decrease of 0.08)
        self.Vh_rp -= 0.08
        self.Vh_rn -= 0.08
        self.Vf_rp -= 0.08
        self.Vf_rn -= 0.08
        self.Vh_n -= 0.08
        self.Vf_n -= 0.08

        # Ensure association strengths don't drop below baseline
        self.Vh_rp = max(self.Vh_rp, self.Vh_rp0)
        self.Vh_rn = max(self.Vh_rn, self.Vh_rn0)
        self.Vf_rp = max(self.Vf_rp, self.Vf_rp0)
        self.Vf_rn = max(self.Vf_rn, self.Vf_rn0)
        self.Vh_n = max(self.Vh_n, self.Vh_n0)
        self.Vf_n = max(self.Vf_n, self.Vf_n0)

        # Calculate synaptic weights from association strengths (sigmoid)
        self.Wh_rp = 1 / (1 + math.exp(-10 * self.Vh_rp + 8))
        self.Wh_rn = 1 / (1 + math.exp(-10 * self.Vh_rn + 8))
        self.Wf_rp = 1 / (1 + math.exp(-10 * self.Vf_rp + 8))
        self.Wf_rn = 1 / (1 + math.exp(-10 * self.Vf_rn + 8))
        self.Wh_n = 1 / (1 + math.exp(-10 * self.Vh_n + 8))
        self.Wf_n = 1 / (1 + math.exp(-10 * self.Vf_n + 8))

        # Check for saturation (NetLogo: threshold 0.83)
        if self.Wh_rp > 0.83:
            self.Wh_rp_saturated = 1
        if self.Wh_rn > 0.83:
            self.Wh_rn_saturated = 1
        if self.Wf_rp > 0.83:
            self.Wf_rp_saturated = 1
        if self.Wf_rn > 0.83:
            self.Wf_rn_saturated = 1
        if self.Wh_n > 0.83:
            self.Wh_n_saturated = 1
        if self.Wf_n > 0.83:
            self.Wf_n_saturated = 1

        # Update dynamic baselines (NetLogo formulas)
        # Baselines decrease when CS is paired with opposite reward
        self.Vh_rp0 = self.Wh_rp_saturated * (0.7 - 0.5 / (1 + math.exp(-5 * (
            self.CS1 * self.R_neg + 0.2 * self.CS1 * self.NR
        ) + 4)))

        self.Vh_rn0 = self.Wh_rn_saturated * (0.7 - 0.5 / (1 + math.exp(-5 * (
            self.CS1 * self.R_pos + 0.2 * self.CS1 * self.NR
        ) + 4)))

        self.Vf_rp0 = self.Wf_rp_saturated * (0.7 - 0.5 / (1 + math.exp(-5 * (
            self.CS2 * self.R_neg + 0.2 * self.CS2 * self.NR
        ) + 4)))

        self.Vf_rn0 = self.Wf_rn_saturated * (0.7 - 0.5 / (1 + math.exp(-5 * (
            self.CS2 * self.R_pos + 0.2 * self.CS2 * self.NR
        ) + 4)))

        self.Vh_n0 = self.Wh_n_saturated * (0.7 - 0.5 / (1 + math.exp(-5 * (
            self.CS1 * self.R_neg + 0.2 * self.CS1 * self.R_pos
        ) + 4)))

        self.Vf_n0 = self.Wf_n_saturated * (0.7 - 0.5 / (1 + math.exp(-5 * (
            self.CS2 * self.R_neg + 0.2 * self.CS2 * self.R_pos
        ) + 4)))

        # Additional CS decay (NetLogo: 0.9 decay for eligibility)
        self.CS1 *= 0.9
        self.CS2 *= 0.9