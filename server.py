"""
server.py - CyberSlug visualization with SOCIAL INTERACTIONS
Shows multiple slugs, biting, and competition
"""
import solara
from model import CyberSlugModel
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


@solara.component
def Page():
    # Model state
    model, set_model = solara.use_state(CyberSlugModel())

    # Parameters
    num_slugs = solara.use_reactive(1)  # NEW: Number of slugs
    hermi_pop = solara.use_reactive(4)
    flab_pop = solara.use_reactive(4)
    fauxflab_pop = solara.use_reactive(4)

    # Selected slug for detailed view
    selected_slug_idx = solara.use_reactive(0)

    # Render trigger
    render_key, set_render_key = solara.use_state(0)

    def reset():
        new_model = CyberSlugModel(
            num_slugs=num_slugs.value,
            hermi_population=hermi_pop.value,
            flab_population=flab_pop.value,
            fauxflab_population=fauxflab_pop.value
        )
        set_model(new_model)
        set_render_key(0)
        selected_slug_idx.set(0)

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
            'hermi': {'color': (0, 255, 255), 'odor': [0.5, 0.5, 0, 0, 0]},
            'flab': {'color': (255, 105, 180), 'odor': [0.5, 0, 0.5, 0, 0]},
            'fauxflab': {'color': (255, 255, 0), 'odor': [0.0, 0.0, 0.5, 0.0, 0]}
        }

        # Adjust each prey type
        for prey_type in ['hermi', 'flab', 'fauxflab']:
            current = current_counts[prey_type]
            target = target_counts[prey_type]

            if current < target:
                # Add new prey
                for _ in range(target - current):
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
        fig = Figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        # Set limits
        ax.set_xlim(0, model.width)
        ax.set_ylim(0, model.height)
        ax.set_aspect('equal')
        ax.set_facecolor('#f0f0f0')
        ax.set_title(f'CyberSlug Social Simulation - Step {model.steps}',
                    fontsize=14, fontweight='bold')

        # Draw all slug paths
        from agents import CyberslugAgent
        for i, slug in enumerate(model.cyberslugs):
            if len(slug.path) > 1:
                path_array = list(slug.path)
                xs, ys = zip(*path_array)
                # Different colors for different slugs
                colors = ['brown', 'darkred', 'darkgreen', 'darkblue', 'purple']
                color = colors[i % len(colors)]
                ax.plot(xs, ys, color=color, linewidth=1, alpha=0.3)

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
                ax.scatter(x, y, c=color, s=100, edgecolors='black',
                          linewidth=1, zorder=3, alpha=0.7)
                prey_counts[agent.prey_type] += 1

        # Draw all slugs with size variation
        for i, slug in enumerate(model.cyberslugs):
            slug_x, slug_y = slug.pos

            # Color based on index
            colors = ['brown', 'darkred', 'darkgreen', 'darkblue', 'purple']
            color = colors[i % len(colors)]

            # Size varies with slug size
            display_size = 400 + (slug.size * 40)

            # Highlight if biting
            if slug.is_biting:
                edgecolor = 'red'
                linewidth = 4
            elif i == selected_slug_idx.value:
                edgecolor = 'gold'
                linewidth = 3
            else:
                edgecolor = 'black'
                linewidth = 2

            ax.scatter(slug_x, slug_y, c=color, s=display_size, marker='o',
                      edgecolors=edgecolor, linewidth=linewidth, zorder=5,
                      alpha=0.8, label=f'Slug {i}' if i < 5 else '')

            # Add slug ID label
            ax.text(slug_x, slug_y, str(i), fontsize=10, fontweight='bold',
                   ha='center', va='center', color='white', zorder=6)

            # Draw direction indicator
            import math
            heading_length = 20 + slug.size
            end_x = slug_x + heading_length * math.cos(math.radians(slug.angle))
            end_y = slug_y + heading_length * math.sin(math.radians(slug.angle))
            ax.arrow(slug_x, slug_y, end_x - slug_x, end_y - slug_y,
                    head_width=10, head_length=8, fc=color, ec=color,
                    zorder=6, alpha=0.7)

            # Show bite indicator
            if slug.is_biting and slug.bite_target:
                tx, ty = slug.bite_target.pos
                ax.plot([slug_x, tx], [slug_y, ty], 'r-', linewidth=3,
                       alpha=0.7, zorder=4)
                ax.text((slug_x + tx)/2, (slug_y + ty)/2, '💥',
                       fontsize=20, ha='center', va='center', zorder=7)

        # Legend
        legend_elements = [
            mpatches.Patch(color='cyan', label=f'Hermissenda ({prey_counts["hermi"]})'),
            mpatches.Patch(color='pink', label=f'Flabellina ({prey_counts["flab"]})'),
            mpatches.Patch(color='yellow', label=f'Faux-Flab ({prey_counts["fauxflab"]})'),
            mlines.Line2D([], [], color='red', linewidth=3,
                         label='Biting!', marker='o', markersize=8),
        ]

        # Add slug legend entries (max 5)
        for i in range(min(5, len(model.cyberslugs))):
            colors = ['brown', 'darkred', 'darkgreen', 'darkblue', 'purple']
            legend_elements.append(
                mpatches.Patch(color=colors[i], label=f'Slug {i}')
            )

        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

        ax.grid(True, alpha=0.2)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')

        return fig

    # Get selected slug
    selected_slug = (model.cyberslugs[selected_slug_idx.value]
                    if selected_slug_idx.value < len(model.cyberslugs)
                    else model.cyberslugs[0] if model.cyberslugs else None)

    # UI Layout
    with solara.Row():
        # Left panel - visualization
        with solara.Column(style={"width": "60%", "padding": "10px"}):
            solara.Markdown("# 🐌🐌 CyberSlug SOCIAL Simulation")
            solara.Info("💡 Multiple slugs compete for prey and can bite each other!")

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
            solara.SliderInt("🐌 Number of Slugs", value=num_slugs, min=1, max=5)
            solara.SliderInt("🔵 Hermissenda", value=hermi_pop, min=0, max=20)
            solara.SliderInt("🔴 Flabellina", value=flab_pop, min=0, max=20)
            solara.SliderInt("🟡 Faux-Flabellina", value=fauxflab_pop, min=0, max=20)

            if selected_slug:
                solara.Markdown("---")
                solara.Markdown(f"## 🔍 Slug {selected_slug_idx.value} Details")

                # Slug selector
                if len(model.cyberslugs) > 1:
                    solara.SliderInt("Select Slug", value=selected_slug_idx,
                                    min=0, max=len(model.cyberslugs)-1)

                solara.Markdown("### 📊 Prey Encounters")
                solara.Text(f"🔵 Hermissenda: {selected_slug.hermi_counter}")
                solara.Text(f"🔴 Flabellina: {selected_slug.flab_counter}")
                solara.Text(f"🟡 Faux-Flabellina: {selected_slug.fauxflab_counter}")

                solara.Markdown("### 💥 Social Interactions")
                solara.Text(f"⚔️ Bites Given: {selected_slug.bite_counter}")
                solara.Text(f"🤕 Times Bitten: {selected_slug.被咬_counter}")
                solara.Text(f"📏 Size: {selected_slug.size:.1f}")
                if selected_slug.is_biting:
                    solara.Success("🔴 BITING NOW!")

                solara.Markdown("---")
                solara.Markdown("## 🧠 Internal States")
                solara.Text(f"Nutrition: {selected_slug.nutrition:.3f}")
                solara.Text(f"Satiation: {selected_slug.satiation:.3f}")
                solara.Text(f"Appetitive State: {selected_slug.app_state:.3f}")
                solara.Text(f"Pain (total): {selected_slug.sns_pain_total:.3f}")
                solara.Text(f"Pain from bites: {selected_slug.pain_from_bite:.3f}")

                solara.Markdown("---")
                solara.Markdown("## 📚 Learning Values")
                solara.Text(f"V_Hermi: {selected_slug.Vh:.3f}")
                solara.Text(f"V_Flab: {selected_slug.Vf:.3f}")
                solara.Text(f"Conspecific W3: {selected_slug.W3_conspecific:.3f}")

            solara.Markdown("---")
            solara.Markdown("## 🌍 Global Stats")
            solara.Text(f"Total Slugs: {len(model.cyberslugs)}")
            solara.Text(f"Total Bites: {sum(s.bite_counter for s in model.cyberslugs)}")
            solara.Text(f"Total Prey Eaten: {sum(s.hermi_counter + s.flab_counter + s.fauxflab_counter for s in model.cyberslugs)}")


page = Page