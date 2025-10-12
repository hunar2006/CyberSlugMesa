"""
model.py - MESA Model for CyberSlug Simulation
Enhanced with ALL NetLogo features:
- Clustering behavior for prey
- Drug odor (4th odor type)
- Immobilize option
- Odor-null mode
- 5 odor types: betaine, hermi, flab, drug, pleur
"""
import numpy as np
import math
from mesa import Model, Agent
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector

try:
    from mesa.time import RandomActivation
except ImportError:
    # MESA 3.0+
    class RandomActivation:
        def __init__(self, model):
            self.model = model
            self.agents = []
            self._agents = {}

        def add(self, agent):
            self.agents.append(agent)
            self._agents[agent.unique_id] = agent

        def remove(self, agent):
            self.agents.remove(agent)
            del self._agents[agent.unique_id]

        def step(self):
            agents = list(self.agents)
            self.model.random.shuffle(agents)
            for agent in agents:
                agent.step()


class CyberSlugModel(Model):
    """
    A model simulating multiple Cyberslugs with ALL NetLogo features.
    - Clustering prey behavior
    - Drug odor support
    - Immobilize mode
    - Odor-null mode (for testing)
    """

    def __init__(self,
                 width=600,
                 height=600,
                 num_slugs=1,
                 hermi_population=4,
                 flab_population=4,
                 fauxflab_population=4,
                 patch_width=200,
                 patch_height=200,
                 clustering=False,
                 cluster_radius=10,
                 immobilize=False,
                 biting=True,
                 odor_null=False,
                 fix_satiation_override=False,
                 fix_satiation_value=1.0):
        super().__init__()

        # Dimensions
        self.width = width
        self.height = height
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.scale = patch_width / width

        # Grid setup - continuous space for movement
        self.space = ContinuousSpace(width, height, torus=True)

        # Scheduler
        self.schedule = RandomActivation(self)

        # Odor patches - 5 types: betaine, hermi, flab, drug, pleur (conspecific)
        self.num_odor_types = 5
        self.patches = np.zeros((self.num_odor_types, patch_width, patch_height))

        # Simulation parameters
        self.prey_radius = 4
        self.sensor_distance = 4
        self.encounter_cooldown = 10
        self.max_slug_size = 20

        # NetLogo features
        self.clustering = clustering
        self.cluster_radius = cluster_radius
        self.immobilize = immobilize
        self.biting = biting
        self.odor_null = odor_null  # If true, slugs don't emit odor
        self.fix_satiation_override = fix_satiation_override
        self.fix_satiation_value = fix_satiation_value

        # Population settings
        self.num_slugs = num_slugs
        self.hermi_population = hermi_population
        self.flab_population = flab_population
        self.fauxflab_population = fauxflab_population

        # Cluster centers (NetLogo style)
        self.hermi_cluster_x = self.random.randrange(width)
        self.hermi_cluster_y = self.random.randrange(height)
        self.flab_cluster_x = self.random.randrange(width)
        self.flab_cluster_y = self.random.randrange(height)
        self.fauxflab_cluster_x = self.random.randrange(width)
        self.fauxflab_cluster_y = self.random.randrange(height)

        # Step counter
        self.ticks = 0
        self.steps = 0

        # Track all slugs
        self.cyberslugs = []

        # For user interactions
        self.being_observed = None  # Selected slug for detailed observation
        self.mouse_x = 0
        self.mouse_y = 0
        self.mouse_down = False

        # Data collector
        self.datacollector = DataCollector(
            model_reporters={
                "Ticks": lambda m: m.ticks,
                "Total_Hermi_Eaten": lambda m: sum([s.hermi_counter for s in m.cyberslugs]),
                "Total_Flab_Eaten": lambda m: sum([s.flab_counter for s in m.cyberslugs]),
                "Total_Fauxflab_Eaten": lambda m: sum([s.fauxflab_counter for s in m.cyberslugs]),
                "Total_Bites": lambda m: sum([s.bite_counter for s in m.cyberslugs]),
                "Avg_Nutrition": lambda m: np.mean([s.nutrition for s in m.cyberslugs]) if m.cyberslugs else 0,
                "Avg_AppState": lambda m: np.mean([s.app_state for s in m.cyberslugs]) if m.cyberslugs else 0,
                "Avg_Vh_rp": lambda m: np.mean([s.Vh_rp for s in m.cyberslugs]) if m.cyberslugs else 0,
                "Avg_Vf_rn": lambda m: np.mean([s.Vf_rn for s in m.cyberslugs]) if m.cyberslugs else 0,
            }
        )

        # Create agents
        self._create_agents()

    def _create_agents(self):
        """Create all agents in the simulation"""
        from agents import CyberslugAgent, PreyAgent

        # Create multiple Cyberslugs
        for i in range(self.num_slugs):
            slug = CyberslugAgent(i, self)
            self.schedule.add(slug)
            # Spread slugs out initially
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            self.space.place_agent(slug, (x, y))
            self.cyberslugs.append(slug)

        # Set first slug as being observed
        self.being_observed = self.cyberslugs[0] if self.cyberslugs else None

        # For backwards compatibility
        self.cyberslug = self.being_observed

        # Create Hermissenda prey (cyan/green)
        base_id = self.num_slugs
        for i in range(self.hermi_population):
            prey = PreyAgent(
                base_id + i,
                self,
                prey_type="hermi",
                color=(0, 255, 255),
                odor=[0.5, 0.5, 0, 0, 0]  # [betaine, hermi, flab, drug, pleur]
            )
            prey.cluster_target = (self.hermi_cluster_x, self.hermi_cluster_y)
            self.schedule.add(prey)

            if self.clustering:
                x = self.hermi_cluster_x + self.random.uniform(-self.cluster_radius, self.cluster_radius)
                y = self.hermi_cluster_y + self.random.uniform(-self.cluster_radius, self.cluster_radius)
            else:
                x = self.random.randrange(self.width)
                y = self.random.randrange(self.height)

            self.space.place_agent(prey, (x, y))

        # Create Flabellina prey (pink/red)
        base_id = self.num_slugs + self.hermi_population
        for i in range(self.flab_population):
            prey = PreyAgent(
                base_id + i,
                self,
                prey_type="flab",
                color=(255, 105, 180),
                odor=[0.5, 0, 0.5, 0, 0]
            )
            prey.cluster_target = (self.flab_cluster_x, self.flab_cluster_y)
            self.schedule.add(prey)

            if self.clustering:
                x = self.flab_cluster_x + self.random.uniform(-self.cluster_radius, self.cluster_radius)
                y = self.flab_cluster_y + self.random.uniform(-self.cluster_radius, self.cluster_radius)
            else:
                x = self.random.randrange(self.width)
                y = self.random.randrange(self.height)

            self.space.place_agent(prey, (x, y))

        # Create Faux-Flabellina prey (blue)
        base_id = self.num_slugs + self.hermi_population + self.flab_population
        for i in range(self.fauxflab_population):
            prey = PreyAgent(
                base_id + i,
                self,
                prey_type="fauxflab",
                color=(255, 255, 0),
                odor=[0.0, 0.0, 0.5, 0.0, 0]  # Only flab odor, no betaine
            )
            prey.cluster_target = (self.fauxflab_cluster_x, self.fauxflab_cluster_y)
            self.schedule.add(prey)

            if self.clustering:
                x = self.fauxflab_cluster_x + self.random.uniform(-self.cluster_radius, self.cluster_radius)
                y = self.fauxflab_cluster_y + self.random.uniform(-self.cluster_radius, self.cluster_radius)
            else:
                x = self.random.randrange(self.width)
                y = self.random.randrange(self.height)

            self.space.place_agent(prey, (x, y))

    def step(self):
        """Advance the model by one step"""
        # Update cluster centers (NetLogo: slow drift)
        if self.clustering:
            self.hermi_cluster_x += self.random.uniform(-0.2, 0.2)
            self.hermi_cluster_y += self.random.uniform(-0.2, 0.2)
            self.flab_cluster_x += self.random.uniform(-0.2, 0.2)
            self.flab_cluster_y += self.random.uniform(-0.2, 0.2)
            self.fauxflab_cluster_x += self.random.uniform(-0.2, 0.2)
            self.fauxflab_cluster_y += self.random.uniform(-0.2, 0.2)

            # Keep clusters in bounds
            self.hermi_cluster_x = self.hermi_cluster_x % self.width
            self.hermi_cluster_y = self.hermi_cluster_y % self.height
            self.flab_cluster_x = self.flab_cluster_x % self.width
            self.flab_cluster_y = self.flab_cluster_y % self.height
            self.fauxflab_cluster_x = self.fauxflab_cluster_x % self.width
            self.fauxflab_cluster_y = self.fauxflab_cluster_y % self.height

        # Update odor patches BEFORE agent steps
        self.update_odor_patches()

        # All agents take their step
        self.schedule.step()

        # Increment step counter
        self.ticks = self.steps

        # Collect data
        self.datacollector.collect(self)

    def update_odor_patches(self):
        """
        NetLogo-style odor dynamics:
        1) Deposit odors from agents
        2) Diffuse to 8 neighbors (Moore neighborhood)
        3) Evaporate
        """
        # Deposit slug odor based on size/nutrition
        for slug in self.cyberslugs:
            x, y = slug.pos
            if self.odor_null:
                # In odor-null mode, slugs emit minimal odor
                self.set_patch_odor(x, y, [0, 0, 0, 0, 0.01])
            else:
                # Normal: emit odor proportional to size
                odor_amount = slug.size / self.max_slug_size
                self.set_patch_odor(x, y, [odor_amount, 0, 0, 0, odor_amount])

        # Diffusion and evaporation
        # NetLogo: hermi, flab, betaine diffuse at 1.0, pleur at 0.5
        diffusion_rates = [1.0, 1.0, 1.0, 1.0, 0.5]  # [betaine, hermi, flab, drug, pleur]
        evap = 0.95  # NetLogo evaporation rate

        for i in range(self.num_odor_types):
            # Diffuse
            amount = diffusion_rates[i]
            field = self.patches[i]
            padded = np.pad(field, 1, mode='wrap')

            neighbors_sum = (
                padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
                padded[1:-1, :-2]                     + padded[1:-1, 2:] +
                padded[2:, :-2]  + padded[2:, 1:-1]  + padded[2:, 2:]
            )

            self.patches[i] = evap * ((1.0 - amount) * field + (amount / 8.0) * neighbors_sum)

    def convert_to_patch_coords(self, x, y):
        """Convert world coordinates to patch grid coordinates"""
        px = int((x - self.width / 2) * self.scale + self.patch_width / 2)
        py = int((y - self.height / 2) * self.scale + self.patch_height / 2)
        px = max(0, min(self.patch_width - 1, px))
        py = max(0, min(self.patch_height - 1, py))
        return px, py

    def set_patch_odor(self, x, y, odorlist):
        """Deposit odor at a given location"""
        px, py = self.convert_to_patch_coords(x, y)
        self.patches[:, px, py] += odorlist

    def get_odor_at_position(self, x, y):
        """Get odor values at a specific position"""
        px, py = self.convert_to_patch_coords(x, y)
        return self.patches[:, px, py].copy()

    def get_sensors(self, x, y, heading):
        """Get sensory input from odor patches based on heading (legacy method)"""
        px, py = self.convert_to_patch_coords(x, y)

        left_x = int(px + self.sensor_distance * math.cos(math.radians(heading - 45)))
        left_y = int(py + self.sensor_distance * math.sin(math.radians(heading - 45)))
        right_x = int(px + self.sensor_distance * math.cos(math.radians(heading + 45)))
        right_y = int(py + self.sensor_distance * math.sin(math.radians(heading + 45)))

        left_x = max(0, min(self.patch_width - 1, left_x))
        left_y = max(0, min(self.patch_height - 1, left_y))
        right_x = max(0, min(self.patch_width - 1, right_x))
        right_y = max(0, min(self.patch_height - 1, right_y))

        return self.patches[:, left_x, left_y], self.patches[:, right_x, right_y]

    def apply_pain_at_position(self, x, y, amount=20.0):
        """Apply pain stimulus at a position (for poker tool)"""
        for slug in self.cyberslugs:
            sx, sy = slug.pos
            dist = math.sqrt((sx - x)**2 + (sy - y)**2)

            if dist < 7 * slug.size:  # Within range
                # Apply pain to all nociceptors based on distance
                for noc in slug.nociceptors:
                    noc_dist = math.sqrt((noc.x - x)**2 + (noc.y - y)**2)
                    noc.painval += amount / (noc_dist + 0.01)

    def set_observed_slug(self, x, y):
        """Set which slug is being observed based on click position"""
        for slug in self.cyberslugs:
            sx, sy = slug.pos
            dist = math.sqrt((sx - x)**2 + (sy - y)**2)
            if dist < 7 * slug.size:
                self.being_observed = slug
                self.cyberslug = slug
                return True
        return False

    def drag_agent(self, x, y):
        """Move agents to mouse position if close enough (dragger tool)"""
        from agents import PreyAgent

        # Try to drag prey
        for agent in self.schedule.agents:
            if isinstance(agent, PreyAgent):
                ax, ay = agent.pos
                dist = math.sqrt((ax - x)**2 + (ay - y)**2)
                if dist < 3:
                    self.space.move_agent(agent, (x, y))
                    return

        # Try to drag slugs
        for slug in self.cyberslugs:
            sx, sy = slug.pos
            dist = math.sqrt((sx - x)**2 + (sy - y)**2)
            if dist < 3:
                self.space.move_agent(slug, (x, y))
                return

    def zero_V_hermi(self):
        """Reset Hermissenda learning values (for testing)"""
        for slug in self.cyberslugs:
            slug.Vh_rp = 0.0
            slug.Vh_rn = 0.0
            slug.Vh_n = 0.0

    def zero_V_flab(self):
        """Reset Flabellina learning values (for testing)"""
        for slug in self.cyberslugs:
            slug.Vf_rp = 0.0
            slug.Vf_rn = 0.0
            slug.Vf_n = 0.0