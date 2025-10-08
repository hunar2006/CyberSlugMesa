"""
server.py - CyberSlug visualization with real-time population control
"""
import solara
from model import CyberSlugModel
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as mpatches


@solara.component
def Page():
    # Model state
    model, set_model = solara.use_state(CyberSlugModel())

    # Parameters
    hermi_pop = solara.use_reactive(4)
    flab_pop = solara.use_reactive(4)
    fauxflab_pop = solara.use_reactive(4)

    # Render trigger
    render_key, set_render_key = solara.use_state(0)

    def reset():
        new_model = CyberSlugModel(
            hermi_population=hermi_pop.value,
            flab_population=flab_pop.value,
            fauxflab_population=fauxflab_pop.value
        )
        set_model(new_model)
        set_render_key(0)

    def do_step():
        # Update populations dynamically before stepping
        update_populations_realtime()
        model.step()
        set_render_key(render_key + 1)

    def do_multiple_steps():
        update_populations_realtime()
        for _ in range(10):
            model.step()
        set_render_key(render_key + 1)

    def update_populations_realtime():
        """Dynamically add or remove prey to match slider values"""
        from agents import PreyAgent

        # Count current prey by type
        current_counts = {'hermi': 0, 'flab': 0, 'fauxflab': 0}
        prey_agents = {'hermi': [], 'flab': [], 'fauxflab': []}

        for agent in list(model.schedule.agents):
            if isinstance(agent, PreyAgent):
                current_counts[agent.prey_type] += 1
                prey_agents[agent.prey_type].append(agent)

        # Target counts from sliders
        target_counts = {
            'hermi': hermi_pop.value,
            'flab': flab_pop.value,
            'fauxflab': fauxflab_pop.value
        }

        # Prey configurations
        prey_config = {
            'hermi': {'color': (0, 255, 255), 'odor': [0.5, 0.5, 0, 0]},
            'flab': {'color': (255, 105, 180), 'odor': [0.5, 0, 0.5, 0]},
            'fauxflab': {'color': (255, 255, 0), 'odor': [0.0, 0.0, 0.5, 0.0]}
        }

        # Adjust each prey type
        for prey_type in ['hermi', 'flab', 'fauxflab']:
            current = current_counts[prey_type]
            target = target_counts[prey_type]

            if current < target:
                # Add new prey
                for _ in range(target - current):
                    # Find next available ID
                    max_id = max([a.unique_id for a in model.schedule.agents]) + 1

                    new_prey = PreyAgent(
                        max_id,
                        model,
                        prey_type=prey_type,
                        color=prey_config[prey_type]['color'],
                        odor=prey_config[prey_type]['odor']
                    )
                    model.schedule.add(new_prey)
                    x = model.random.randrange(model.width)
                    y = model.random.randrange(model.height)
                    model.space.place_agent(new_prey, (x, y))

            elif current > target:
                # Remove excess prey
                to_remove = current - target
                for agent in prey_agents[prey_type][:to_remove]:
                    model.space.remove_agent(agent)
                    model.schedule.remove(agent)

        # Update model's population tracking
        model.hermi_population = hermi_pop.value
        model.flab_population = flab_pop.value
        model.fauxflab_population = fauxflab_pop.value

    # Create visualization
    def create_plot():
        fig = Figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        # Set limits
        ax.set_xlim(0, model.width)
        ax.set_ylim(0, model.height)
        ax.set_aspect('equal')
        ax.set_facecolor('#f0f0f0')
        ax.set_title(f'CyberSlug Simulation - Step {model.steps}', fontsize=14, fontweight='bold')

        # Draw slug path
        if len(model.cyberslug.path) > 1:
            path_array = list(model.cyberslug.path)
            xs, ys = zip(*path_array)
            ax.plot(xs, ys, 'brown', linewidth=1, alpha=0.5, label='Slug Trail')

        # Draw prey
        from agents import PreyAgent
        prey_counts = {'hermi': 0, 'flab': 0, 'fauxflab': 0}
        for agent in model.schedule.agents:
            if isinstance(agent, PreyAgent):
                x, y = agent.pos
                color_map = {
                    'hermi': 'cyan',
                    'flab': 'pink',
                    'fauxflab': 'yellow'
                }
                color = color_map.get(agent.prey_type, 'gray')
                ax.scatter(x, y, c=color, s=100, edgecolors='black', linewidth=1, zorder=3)
                prey_counts[agent.prey_type] += 1

        # Draw slug (larger circle)
        slug_x, slug_y = model.cyberslug.pos
        ax.scatter(slug_x, slug_y, c='brown', s=800, marker='o',
                  edgecolors='black', linewidth=2, zorder=5, label='Cyberslug')

        # Add direction indicator
        import math
        heading_length = 30
        end_x = slug_x + heading_length * math.cos(math.radians(model.cyberslug.angle))
        end_y = slug_y + heading_length * math.sin(math.radians(model.cyberslug.angle))
        ax.arrow(slug_x, slug_y, end_x - slug_x, end_y - slug_y,
                head_width=15, head_length=10, fc='darkred', ec='darkred', zorder=6)

        # Legend with actual counts
        legend_elements = [
            mpatches.Patch(color='cyan', label=f'Hermissenda ({prey_counts["hermi"]})'),
            mpatches.Patch(color='pink', label=f'Flabellina ({prey_counts["flab"]})'),
            mpatches.Patch(color='yellow', label=f'Faux-Flab ({prey_counts["fauxflab"]})'),
            mpatches.Patch(color='brown', label='Cyberslug')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

        ax.grid(True, alpha=0.2)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')

        return fig

    # UI Layout
    with solara.Row():
        # Left panel - visualization
        with solara.Column(style={"width": "60%", "padding": "10px"}):
            solara.Markdown("# 🐌 CyberSlug Simulation")
            solara.Info("💡 Adjust sliders and click Step to see changes in real-time!")

            # Visualization
            fig = create_plot()
            solara.FigureMatplotlib(fig, dependencies=[render_key])

        # Right panel - controls and stats
        with solara.Column(style={"width": "40%", "padding": "10px"}):
            solara.Markdown(f"### Step: {model.steps}")

            # Control buttons
            with solara.Row():
                solara.Button("🔄 Reset", on_click=reset, color="primary")
                solara.Button("▶️ Step", on_click=do_step, color="success")
                solara.Button("⏩ Step 10x", on_click=do_multiple_steps, color="warning")

            solara.Markdown("---")
            solara.Markdown("## 🎛️ Population Settings")
            solara.Markdown("*Changes apply on next step*")
            solara.SliderInt("🔵 Hermissenda", value=hermi_pop, min=0, max=20)
            solara.SliderInt("🔴 Flabellina", value=flab_pop, min=0, max=20)
            solara.SliderInt("🟡 Faux-Flabellina", value=fauxflab_pop, min=0, max=20)

            solara.Markdown("---")
            solara.Markdown("## 📊 Prey Encounters")
            solara.Text(f"🔵 Hermissenda Eaten: {model.cyberslug.hermi_counter}")
            solara.Text(f"🔴 Flabellina Eaten: {model.cyberslug.flab_counter}")
            solara.Text(f"🟡 Faux-Flabellina Eaten: {model.cyberslug.fauxflab_counter}")

            solara.Markdown("---")
            solara.Markdown("## 🧠 Internal States")
            solara.Text(f"Nutrition: {model.cyberslug.nutrition:.3f}")
            solara.Text(f"Appetitive State: {model.cyberslug.app_state:.3f}")
            solara.Text(f"Incentive: {model.cyberslug.incentive:.3f}")
            solara.Text(f"Somatic Map: {model.cyberslug.somatic_map:.3f}")

            solara.Markdown("---")
            solara.Markdown("## 📚 Learning Values")
            solara.Text(f"V_Hermi: {model.cyberslug.Vh:.3f}")
            solara.Text(f"V_Flab: {model.cyberslug.Vf:.3f}")


page = Page