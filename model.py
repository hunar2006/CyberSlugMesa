"""
model.py - MESA Model for CyberSlug Simulation
Core model implementing the environment and stepping logic
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
    A model simulating a Cyberslug learning to navigate using chemosensory cues.
    Includes prey (Hermissenda, Flabellina, Faux-Flabellina) with odor trails.
    """

    def __init__(self, width=600, height=600,
                 hermi_population=4, flab_population=4, fauxflab_population=4,
                 patch_width=200, patch_height=200):
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

        # Odor patches - 4 types: betaine, hermi, flab, drug
        self.num_odor_types = 4
        self.patches = np.zeros((self.num_odor_types, patch_width, patch_height))

        # Simulation parameters
        self.prey_radius = 4
        self.sensor_distance = 4
        self.encounter_cooldown = 10

        # Population settings
        self.hermi_population = hermi_population
        self.flab_population = flab_population
        self.fauxflab_population = fauxflab_population

        # Step counter
        self.ticks = 0
        self.steps = 0

        # Data collector
        self.datacollector = DataCollector(
            model_reporters={
                "Ticks": lambda m: m.ticks,
                "Hermi_Eaten": lambda m: m.cyberslug.hermi_counter,
                "Flab_Eaten": lambda m: m.cyberslug.flab_counter,
                "Fauxflab_Eaten": lambda m: m.cyberslug.fauxflab_counter,
                "Appetitive_State": lambda m: m.cyberslug.app_state,
                "Incentive": lambda m: m.cyberslug.incentive,
                "Nutrition": lambda m: m.cyberslug.nutrition,
            }
        )

        # Create agents
        self._create_agents()

    def _create_agents(self):
        """Create all agents in the simulation"""
        from agents import CyberslugAgent, PreyAgent

        # Create the Cyberslug (unique_id=0)
        self.cyberslug = CyberslugAgent(0, self)
        self.schedule.add(self.cyberslug)
        self.space.place_agent(self.cyberslug, (self.width/2, self.height/2))

        # Create Hermissenda prey (cyan)
        for i in range(self.hermi_population):
            prey = PreyAgent(
                i + 1,
                self,
                prey_type="hermi",
                color=(0, 255, 255),
                odor=[0.5, 0.5, 0, 0]
            )
            self.schedule.add(prey)
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            self.space.place_agent(prey, (x, y))

        # Create Flabellina prey (pink)
        base_id = self.hermi_population + 1
        for i in range(self.flab_population):
            prey = PreyAgent(
                base_id + i,
                self,
                prey_type="flab",
                color=(255, 105, 180),
                odor=[0.5, 0, 0.5, 0]
            )
            self.schedule.add(prey)
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            self.space.place_agent(prey, (x, y))

        # Create Faux-Flabellina prey (yellow)
        base_id = self.hermi_population + self.flab_population + 1
        for i in range(self.fauxflab_population):
            prey = PreyAgent(
                base_id + i,
                self,
                prey_type="fauxflab",
                color=(255, 255, 0),
                odor=[0.0, 0.0, 0.5, 0.0]
            )
            self.schedule.add(prey)
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            self.space.place_agent(prey, (x, y))

    def step(self):
        """Advance the model by one step"""
        # DON'T increment - MESA does it automatically!
        # self.steps += 1  <-- REMOVE THIS LINE

        self.ticks = self.steps  # Just sync ticks with steps

        # Update odor patches
        self.update_odor_patches()

        # All agents take their step
        for agent in list(self.schedule.agents):
            agent.step()

        # Collect data
        self.datacollector.collect(self)

    def update_odor_patches(self):
        """
        NetLogo-style odor dynamics:
        1) Diffuse to 8 neighbors (Moore neighborhood)
        2) Evaporate
        3) Using toroidal boundaries
        """
        amount = 0.5
        evap = 0.95

        for i in range(self.patches.shape[0]):
            field = self.patches[i]
            padded = np.pad(field, 1, mode='wrap')

            neighbors_sum = (
                padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
                padded[1:-1, :-2]                     + padded[1:-1, 2:] +
                padded[2:, :-2]  + padded[2:, 1:-1]  + padded[2:, 2:]
            )

            self.patches[i] = evap * ((1.0 - amount) * field + (amount / 8.0) * neighbors_sum)

    def convert_patch_to_coord(self, x, y):
        """Convert world coordinates to patch grid coordinates"""
        px = int((x - self.width / 2) * self.scale + self.patch_width / 2)
        py = int((y - self.height / 2) * self.scale + self.patch_height / 2)
        px = max(0, min(self.patch_width - 1, px))
        py = max(0, min(self.patch_height - 1, py))
        return px, py

    def set_patch_odor(self, x, y, odorlist):
        """Deposit odor at a given location"""
        px, py = self.convert_patch_to_coord(x, y)
        self.patches[:, px, py] += odorlist

    def get_sensors(self, x, y, heading):
        """Get sensory input from odor patches based on heading"""
        px, py = self.convert_patch_to_coord(x, y)

        left_x = int(px + self.sensor_distance * math.cos(math.radians(heading - 45)))
        left_y = int(py + self.sensor_distance * math.sin(math.radians(heading - 45)))
        right_x = int(px + self.sensor_distance * math.cos(math.radians(heading + 45)))
        right_y = int(py + self.sensor_distance * math.sin(math.radians(heading + 45)))

        left_x = max(0, min(self.patch_width - 1, left_x))
        left_y = max(0, min(self.patch_height - 1, left_y))
        right_x = max(0, min(self.patch_width - 1, right_x))
        right_y = max(0, min(self.patch_height - 1, right_y))

        return self.patches[:, left_x, left_y], self.patches[:, right_x, right_y]