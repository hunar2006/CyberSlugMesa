"""
server.py - Simple web server for CyberSlug using Solara
"""
import solara
from mesa.visualization import SolaraViz, make_space_component
from model import CyberSlugModel
from agents import CyberslugAgent, PreyAgent


def agent_portrayal(agent):
    """Define how agents are displayed"""
    if isinstance(agent, CyberslugAgent):
        return {
            "color": "brown",
            "size": 50,
        }
    elif isinstance(agent, PreyAgent):
        # Convert RGB tuple to hex color
        r, g, b = agent.color
        color = f"#{r:02x}{g:02x}{b:02x}"

        return {
            "color": color,
            "size": 10,
        }
    return {}


# Model parameters
model_params = {
    "hermi_population": {
        "type": "SliderInt",
        "value": 4,
        "label": "Hermissenda Population",
        "min": 0,
        "max": 20,
        "step": 1,
    },
    "flab_population": {
        "type": "SliderInt",
        "value": 4,
        "label": "Flabellina Population",
        "min": 0,
        "max": 20,
        "step": 1,
    },
    "fauxflab_population": {
        "type": "SliderInt",
        "value": 4,
        "label": "Faux-Flabellina Population",
        "min": 0,
        "max": 20,
        "step": 1,
    },
}


# Create visualization components
def make_plot(measure):
    def get_measure(model):
        return getattr(model.datacollector.model_vars[measure], "__iter__", lambda: [])()

    return solara.FigureMatplotlib(label=measure)


# Create the page
page = SolaraViz(
    CyberSlugModel,
    components=[
        make_space_component(agent_portrayal),
    ],
    model_params=model_params,
    name="CyberSlug Simulation",
)