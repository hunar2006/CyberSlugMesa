"""
server.py - Simple visualization for CyberSlug
"""
import solara
from model import CyberSlugModel


# Global model to avoid render loops
model_store = solara.reactive(CyberSlugModel())


@solara.component
def Page():
    # Parameters
    hermi_pop = solara.use_reactive(4)
    flab_pop = solara.use_reactive(4)
    fauxflab_pop = solara.use_reactive(4)

    # Force render flag
    force_update, set_force_update = solara.use_state(0)

    def reset():
        model_store.value = CyberSlugModel(
            hermi_population=hermi_pop.value,
            flab_population=flab_pop.value,
            fauxflab_population=fauxflab_pop.value
        )
        set_force_update(force_update + 1)

    def do_step():
        model_store.value.step()
        set_force_update(force_update + 1)

    model = model_store.value

    # UI Layout
    with solara.Column():
        solara.Markdown("# CyberSlug Simulation")

        solara.Markdown(f"### Step: {model.steps}")

        with solara.Row():
            solara.Button("Reset", on_click=reset)
            solara.Button("Step", on_click=do_step)

        solara.Markdown("---")
        solara.Markdown("## Population Settings")
        solara.SliderInt("Hermissenda", value=hermi_pop, min=0, max=20)
        solara.SliderInt("Flabellina", value=flab_pop, min=0, max=20)
        solara.SliderInt("Faux-Flabellina", value=fauxflab_pop, min=0, max=20)

        solara.Markdown("---")
        solara.Markdown("## Prey Encounters")
        solara.Text(f"Hermissenda Eaten: {model.cyberslug.hermi_counter}")
        solara.Text(f"Flabellina Eaten: {model.cyberslug.flab_counter}")
        solara.Text(f"Faux-Flabellina Eaten: {model.cyberslug.fauxflab_counter}")

        solara.Markdown("---")
        solara.Markdown("## Internal States")
        solara.Text(f"Nutrition: {model.cyberslug.nutrition:.3f}")
        solara.Text(f"Appetitive State: {model.cyberslug.app_state:.3f}")
        solara.Text(f"Incentive: {model.cyberslug.incentive:.3f}")

        solara.Markdown("---")
        solara.Markdown("## Learning Values")
        solara.Text(f"V_Hermi: {model.cyberslug.Vh:.3f}")
        solara.Text(f"V_Flab: {model.cyberslug.Vf:.3f}")


page = Page