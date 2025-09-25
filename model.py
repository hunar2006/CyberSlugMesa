import mesa
import numpy as np
from scipy.ndimage import gaussian_filter
from agents import CyberslugAgent, HermiAgent, FlabAgent, FauxFlabAgent
from config import *


class CyberslugModel(mesa.Model):
    """
    Mesa model implementing Cyberslug ecosystem with odor diffusion
    """

    def __init__(self, hermi_count=HERMI_POPULATION, flab_count=FLAB_POPULATION,
                 fauxflab_count=FAUXFLAB_POPULATION):
        super().__init__()

        # Model parameters
        self.hermi_count = hermi_count
        self.flab_count = flab_count
        self.fauxflab_count = fauxflab_count

        # Create grid and scheduler
        self.grid = mesa.space.MultiGrid(GRID_WIDTH, GRID_HEIGHT, TORUS_SPACE)
        self.schedule = mesa.time.RandomActivation(self)

        # Odor diffusion system (4 channels)
        self.odor_patches = np.zeros((4, GRID_WIDTH, GRID_HEIGHT))

        # Create agents
        self.create_agents()

        # Data collection
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Cyberslug_Nutrition": lambda m: m.get_cyberslug().nutrition,
                "Cyberslug_AppState": lambda m: m.get_cyberslug().app_state,
                "Cyberslug_Incentive": lambda m: m.get_cyberslug().incentive,
                "Hermi_Encounters": lambda m: m.get_cyberslug().hermi_counter,
                "Flab_Encounters": lambda m: m.get_cyberslug().flab_counter,
                "Drug_Encounters": lambda m: m.get_cyberslug().drug_counter,
                "Somatic_Map": lambda m: m.get_cyberslug().somatic_map,
            },
            agent_reporters={
                "x": "pos[0]" if "pos" in dir(mesa.Agent) else lambda a: a.pos[0] if a.pos else 0,
                "y": "pos[1]" if "pos" in dir(mesa.Agent) else lambda a: a.pos[1] if a.pos else 0,
            }
        )

        self.running = True

    def create_agents(self):
        """Create all agents in the model"""
        agent_id = 0

        # Create Cyberslug
        cyberslug = CyberslugAgent(agent_id, self)
        self.schedule.add(cyberslug)
        pos = self.grid.find_empty()
        self.grid.place_agent(cyberslug, pos)
        agent_id += 1

        # Create Hermi prey
        for _ in range(self.hermi_count):
            hermi = HermiAgent(agent_id, self)
            self.schedule.add(hermi)
            pos = self.grid.find_empty()
            if pos:
                self.grid.place_agent(hermi, pos)
            agent_id += 1

        # Create Flab prey
        for _ in range(self.flab_count):
            flab = FlabAgent(agent_id, self)
            self.schedule.add(flab)
            pos = self.grid.find_empty()
            if pos:
                self.grid.place_agent(flab, pos)
            agent_id += 1

        # Create FauxFlab prey
        for _ in range(self.fauxflab_count):
            fauxflab = FauxFlabAgent(agent_id, self)
            self.schedule.add(fauxflab)
            pos = self.grid.find_empty()
            if pos:
                self.grid.place_agent(fauxflab, pos)
            agent_id += 1

    def get_cyberslug(self):
        """Get the Cyberslug agent"""
        for agent in self.schedule.agents:
            if isinstance(agent, CyberslugAgent):
                return agent
        return None

    def deposit_odor(self, pos, odor_signature):
        """Deposit odor at position"""
        x, y = pos
        if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
            for i, concentration in enumerate(odor_signature):
                self.odor_patches[i, x, y] += concentration

    def get_odor_at_position(self, x, y):
        """Get odor concentrations at position"""
        x = int(max(0, min(GRID_WIDTH - 1, x)))
        y = int(max(0, min(GRID_HEIGHT - 1, y)))
        return self.odor_patches[:, x, y].copy()

    def update_odors(self):
        """Apply diffusion and decay to odors"""
        for i in range(4):
            self.odor_patches[i] = gaussian_filter(self.odor_patches[i], sigma=1) * 0.95

    def step(self):
        """Advance model by one step"""
        self.schedule.step()
        self.update_odors()
        self.datacollector.collect(self)
