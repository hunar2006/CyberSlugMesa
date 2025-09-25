import mesa
from mesa.visualization import SolaraViz, make_space_component, make_plot_component
from model import CyberslugModel
from config import COLORS
from agents import CyberslugAgent, HermiAgent, FlabAgent, FauxFlabAgent

def agent_portrayal(agent):
    """Define how agents appear in visualization"""
    if isinstance(agent, CyberslugAgent):
        return {
            "color": COLORS['cyberslug'],
            "size": 3,
            "shape": "circle",
            "layer": 2,
        }
    elif isinstance(agent, HermiAgent):
        return {
            "color": COLORS['hermi'],
            "size": 2,
            "shape": "circle",
            "layer": 1,
        }
    elif isinstance(agent, FlabAgent):
        return {
            "color": COLORS['flab'],
            "size": 2,
            "shape": "circle",
            "layer": 1,
        }
    elif isinstance(agent, FauxFlabAgent):
        return {
            "color": COLORS['fauxflab'],
            "size": 2,
            "shape": "circle",
            "layer": 1,
        }

model_params = {
    "hermi_count": {
        "type": "SliderInt",
        "value": 4,
        "label": "Hermi Population:",
        "min": 0,
        "max": 20,
        "step": 1,
    },
    "flab_count": {
        "type": "SliderInt",
        "value": 4,
        "label": "Flab Population:",
        "min": 0,
        "max": 20,
        "step": 1,
    },
    "fauxflab_count": {
        "type": "SliderInt",
        "value": 4,
        "label": "FauxFlab Population:",
        "min": 0,
        "max": 20,
        "step": 1,
    },
}

# Create visualization
page = SolaraViz(
    CyberslugModel,
    components=[
        make_space_component(agent_portrayal),
        make_plot_component("Cyberslug_Nutrition"),
        make_plot_component("Cyberslug_AppState"),
        make_plot_component("Cyberslug_Incentive"),
        make_plot_component(["Hermi_Encounters", "Flab_Encounters", "Drug_Encounters"]),
    ],
    model_params=model_params,
    name="Cyberslug Neural Simulation"
)
