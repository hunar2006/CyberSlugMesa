from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import numpy as np
from scipy.ndimage import gaussian_filter
import random
import math
from agents import CyberslugAgent, HermiAgent, FlabAgent, FauxFlabAgent
from config import *


class CyberslugModel(Model):
    def __init__(self, hermi_count=4, flab_count=4, fauxflab_count=4):
        super().__init__()

        self.hermi_count = hermi_count
        self.flab_count = flab_count
        self.fauxflab_count = fauxflab_count

        # Create grid and scheduler
        self.grid = MultiGrid(GRID_WIDTH, GRID_HEIGHT, True)
        self.schedule = RandomActivation(self)

        # Initialize odor patch system (EXACT ORIGINAL)
        self.patches = [np.zeros((PATCH_WIDTH, PATCH_HEIGHT)) for _ in range(NUM_ODOR_TYPES)]

        # Create agents
        self.create_agents()

        # Data collection with all original variables
        self.datacollector = DataCollector(
            model_reporters={
                "Cyberslug_Nutrition": lambda m: m.get_cyberslug().nutrition if m.get_cyberslug() else 0,
                "Cyberslug_AppState": lambda m: m.get_cyberslug().app_state if m.get_cyberslug() else 0,
                "Cyberslug_Incentive": lambda m: m.get_cyberslug().incentive if m.get_cyberslug() else 0,
                "Cyberslug_Satiation": lambda m: m.get_cyberslug().satiation if m.get_cyberslug() else 0,
                "Cyberslug_Pain": lambda m: m.get_cyberslug().pain if m.get_cyberslug() else 0,
                "Cyberslug_SomaticMap": lambda m: m.get_cyberslug().somatic_map if m.get_cyberslug() else 0,
                "Hermi_Encounters": lambda m: m.get_cyberslug().hermi_counter if m.get_cyberslug() else 0,
                "Flab_Encounters": lambda m: m.get_cyberslug().flab_counter if m.get_cyberslug() else 0,
                "Drug_Encounters": lambda m: m.get_cyberslug().drug_counter if m.get_cyberslug() else 0,
                "Reward_Positive": lambda m: m.get_cyberslug().reward_pos if m.get_cyberslug() else 0,
                "Reward_Negative": lambda m: m.get_cyberslug().reward_neg if m.get_cyberslug() else 0,
            }
        )

        self.running = True

    def create_agents(self):
        agent_id = 0

        # Create Cyberslug at center
        cyberslug = CyberslugAgent(agent_id, self)
        self.schedule.add(cyberslug)
        pos = (GRID_WIDTH // 2, GRID_HEIGHT // 2)
        self.grid.place_agent(cyberslug, pos)
        agent_id += 1

        # Create prey agents
        for _ in range(self.hermi_count):
            hermi = HermiAgent(agent_id, self)
            self.schedule.add(hermi)
            pos = (random.randint(50, GRID_WIDTH - 50), random.randint(50, GRID_HEIGHT - 50))
            self.grid.place_agent(hermi, pos)
            agent_id += 1

        for _ in range(self.flab_count):
            flab = FlabAgent(agent_id, self)
            self.schedule.add(flab)
            pos = (random.randint(50, GRID_WIDTH - 50), random.randint(50, GRID_HEIGHT - 50))
            self.grid.place_agent(flab, pos)
            agent_id += 1

        for _ in range(self.fauxflab_count):
            fauxflab = FauxFlabAgent(agent_id, self)
            self.schedule.add(fauxflab)
            pos = (random.randint(50, GRID_WIDTH - 50), random.randint(50, GRID_HEIGHT - 50))
            self.grid.place_agent(fauxflab, pos)
            agent_id += 1

    def get_cyberslug(self):
        for agent in self.schedule.agents:
            if isinstance(agent, CyberslugAgent):
                return agent
        return None

    def convert_patch_to_coord(self, x, y):
        """EXACT ORIGINAL coordinate conversion"""
        px = int((x - GRID_WIDTH / 2) * SCALE + PATCH_WIDTH / 2)
        py = int((y - GRID_HEIGHT / 2) * SCALE + PATCH_HEIGHT / 2)
        px = max(0, min(PATCH_WIDTH - 1, px))
        py = max(0, min(PATCH_HEIGHT - 1, py))
        return px, py

    def get_odor_at_position(self, x, y):
        """Get odor concentrations at position using patch system"""
        px, py = self.convert_patch_to_coord(x, y)
        return [self.patches[i][px, py] for i in range(NUM_ODOR_TYPES)]

    def set_patch(self, x, y, odor_list):
        """EXACT ORIGINAL odor deposition"""
        px, py = self.convert_patch_to_coord(x, y)
        for i in range(NUM_ODOR_TYPES):
            self.patches[i][px, py] += odor_list[i]

    def update_odors(self):
        """EXACT ORIGINAL odor diffusion and decay"""
        for i in range(NUM_ODOR_TYPES):
            self.patches[i] = gaussian_filter(self.patches[i], sigma=1) * 0.95

    def step(self):
        """Step model with exact original sequence"""
        self.schedule.step()
        self.update_odors()
        self.datacollector.collect(self)
