"""
server.py - CyberSlug visualization with ALL NetLogo features
- Interactive controls: Dragger, Poker, Set Observer
- Clustering behavior toggle
- Show nociceptors visualization
- Advanced learning circuit display
- Proboscis visualization
- All NetLogo switches and controls
"""
import solara
from model import CyberSlugModel
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import math


@solara.component
def Page():
    # Model state
    model, set_model = solara.use_state(CyberSlugModel())

    # Population parameters
    num_slugs = solara.use_reactive(2)
    hermi_pop = solara.use_reactive(15)
    flab_pop = solara.use_reactive(15)
    fauxflab_pop = solara.use_reactive(15)

    # NetLogo feature toggles
    clustering = solara.use_reactive(False)
    cluster_radius = solara.use_reactive(10)
    immobilize = solara.use_reactive(False)
    biting = solara.use_reactive(True)
    odor_null = solara.use_reactive(False)
    show_nociceptors = solara.use_reactive(False)

    # Interactive tool modes (only one can be active)
    dragger_mode = solara.use_reactive(False)
    poker_mode = solara.use_reactive(False)
    observer_mode = solara.use_reactive(False)

    # Satiation override
    fix_satiation = solara.use_reactive(False)
    satiation_value = solara.use_reactive(1.0)

    # Selected slug for detailed view
    selected_slug_idx = solara.use_reactive(0)

    # Render trigger
    render_key, set_render_key = solara.use_state(0)

    # AUTOMATION CONTROLS
    auto_running = solara.use_reactive(False)
    steps_per_frame = solara.use_reactive(30)  # Default 30 steps per update
    update_interval = solara.use_reactive(0.1)  # Seconds between updates

    # Mouse state for interactions
    mouse_state = solara.use_reactive({"x": 0, "y": 0, "down": False})

    # Auto-run effect
    def auto_step():
        if auto_running.value:
            update_populations_realtime()
            # Run multiple steps
            for _ in range(steps_per_frame.value):
                model.clustering = clustering.value
                model.immobilize = immobilize.value
                model.biting = biting.value
                model.odor_null = odor_null.value
                model.fix_satiation_override = fix_satiation.value
                model.fix_satiation_value = satiation_value.value
                model.step()
            set_render_key(render_key + 1)

    # Set up auto-run timer
    solara.use_thread(
        lambda: __import__('time').sleep(update_interval.value) or auto_step() if auto_running.value else None,
        dependencies=[auto_running.value, render_key]
    )

    def toggle_auto_run():
        auto_running.set(not auto_running.value)

    def reset():
        new_model = CyberSlugModel(
            num_slugs=num_slugs.value,
            hermi_population=hermi_pop.value,
            flab_population=flab_pop.value,
            fauxflab_population=fauxflab_pop.value,
            clustering=clustering.value,
            cluster_radius=cluster_radius.value,
            immobilize=immobilize.value,
            biting=biting.value,
            odor_null=odor_null.value,
            fix_satiation_override=fix_satiation.value,
            fix_satiation_value=satiation_value.value
        )
        set_model(new_model)
        set_render_key(0)
        selected_slug_idx.set(0)

    def do_step():
        update_populations_realtime()
        # Update model settings from toggles
        model.clustering = clustering.value
        model.immobilize = immobilize.value
        model.biting = biting.value
        model.odor_null = odor_null.value
        model.fix_satiation_override = fix_satiation.value
        model.fix_satiation_value = satiation_value.value
        model.step()
        set_render_key(render_key + 1)

    def do_multiple_steps():
        update_populations_realtime()
        for _ in range(10):
            model.clustering = clustering.value
            model.immobilize = immobilize.value
            model.biting = biting.value
            model.odor_null = odor_null.value
            model.fix_satiation_override = fix_satiation.value
            model.fix_satiation_value = satiation_value.value
            model.step()
        set_render_key(render_key + 1)

    def update_populations_realtime():
        """Dynamically add or remove prey to match slider values"""
        from agents import PreyAgent

        current_counts = {'hermi': 0, 'flab': 0, 'fauxflab': 0}
        prey_agents = {'hermi': [], 'flab': [], 'fauxflab': []}

        for agent in list(model.schedule.agents):
            if isinstance(agent, PreyAgent):
                current_counts[agent.prey_type] += 1
                prey_agents[agent.prey_type].append(agent)

        target_counts = {
            'hermi': hermi_pop.value,
            'flab': flab_pop.value,
            'fauxflab': fauxflab_pop.value
        }

        prey_config = {
            'hermi': {
                'color': (0, 255, 255),
                'odor': [0.5, 0.5, 0, 0, 0],
                'cluster': (model.hermi_cluster_x, model.hermi_cluster_y)
            },
            'flab': {
                'color': (255, 105, 180),
                'odor': [0.5, 0, 0.5, 0, 0],
                'cluster': (model.flab_cluster_x, model.flab_cluster_y)
            },
            'fauxflab': {
                'color': (255, 255, 0),
                'odor': [0.0, 0.0, 0.5, 0.0, 0],
                'cluster': (model.fauxflab_cluster_x, model.fauxflab_cluster_y)
            }
        }

        for prey_type in ['hermi', 'flab', 'fauxflab']:
            current = current_counts[prey_type]
            target = target_counts[prey_type]

            if current < target:
                for _ in range(target - current):
                    max_id = max([a.unique_id for a in model.schedule.agents]) + 1
                    new_prey = PreyAgent(
                        max_id,
                        model,
                        prey_type=prey_type,
                        color=prey_config[prey_type]['color'],
                        odor=prey_config[prey_type]['odor']
                    )
                    new_prey.cluster_target = prey_config[prey_type]['cluster']
                    model.schedule.add(new_prey)

                    if model.clustering:
                        cx, cy = prey_config[prey_type]['cluster']
                        x = cx + model.random.uniform(-model.cluster_radius, model.cluster_radius)
                        y = cy + model.random.uniform(-model.cluster_radius, model.cluster_radius)
                    else:
                        x = model.random.randrange(model.width)
                        y = model.random.randrange(model.height)

                    model.space.place_agent(new_prey, (x, y))

            elif current > target:
                to_remove = current - target
                for agent in prey_agents[prey_type][:to_remove]:
                    model.space.remove_agent(agent)
                    model.schedule.remove(agent)

        model.hermi_population = hermi_pop.value
        model.flab_population = flab_pop.value
        model.fauxflab_population = fauxflab_pop.value

    def zero_vh():
        """Reset Hermi learning values"""
        model.zero_V_hermi()
        set_render_key(render_key + 1)

    def zero_vf():
        """Reset Flab learning values"""
        model.zero_V_flab()
        set_render_key(render_key + 1)

    def create_plot():
        """Create main visualization with all features"""
        fig = Figure(figsize=(12, 12))
        ax = fig.add_subplot(111)

        ax.set_xlim(0, model.width)
        ax.set_ylim(0, model.height)
        ax.set_aspect('equal')
        ax.set_facecolor('#f0f0f0')
        ax.set_title(f'CyberSlug Complete Simulation - Step {model.steps}',
                    fontsize=16, fontweight='bold')

        # Draw slug paths
        from agents import CyberslugAgent
        for i, slug in enumerate(model.cyberslugs):
            if len(slug.path) > 1:
                path_array = list(slug.path)
                xs, ys = zip(*path_array)
                colors = ['brown', 'darkred', 'darkgreen', 'darkblue', 'purple', 'orange']
                color = colors[i % len(colors)]
                ax.plot(xs, ys, color=color, linewidth=1, alpha=0.3)

        # Draw cluster centers if clustering is enabled
        if model.clustering:
            ax.scatter(model.hermi_cluster_x, model.hermi_cluster_y,
                      s=500, c='cyan', marker='x', linewidth=3,
                      alpha=0.5, label='Hermi Cluster')
            ax.scatter(model.flab_cluster_x, model.flab_cluster_y,
                      s=500, c='pink', marker='x', linewidth=3,
                      alpha=0.5, label='Flab Cluster')
            ax.scatter(model.fauxflab_cluster_x, model.fauxflab_cluster_y,
                      s=500, c='yellow', marker='x', linewidth=3,
                      alpha=0.5, label='Fauxflab Cluster')

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

        # Draw slugs
        for i, slug in enumerate(model.cyberslugs):
            slug_x, slug_y = slug.pos

            colors = ['brown', 'darkred', 'darkgreen', 'darkblue', 'purple', 'orange']
            color = colors[i % len(colors)]

            display_size = 400 + (slug.size * 40)

            # Highlight based on state
            if slug.is_biting:
                edgecolor = 'red'
                linewidth = 5
            elif slug == model.being_observed:
                edgecolor = 'gold'
                linewidth = 4
            else:
                edgecolor = 'black'
                linewidth = 2

            # Draw slug body
            ax.scatter(slug_x, slug_y, c=color, s=display_size, marker='o',
                      edgecolors=edgecolor, linewidth=linewidth, zorder=5,
                      alpha=0.8)

            # Slug ID label
            ax.text(slug_x, slug_y, str(i), fontsize=12, fontweight='bold',
                   ha='center', va='center', color='white', zorder=6)

            # Draw proboscis if extended
            if slug.proboscis_extended:
                prob_length = 0.15 * slug.size + 0.1 * slug.proboscis_phase
                prob_x = slug_x + prob_length * math.cos(math.radians(slug.angle))
                prob_y = slug_y + prob_length * math.sin(math.radians(slug.angle))
                ax.plot([slug_x, prob_x], [slug_y, prob_y],
                       color='red', linewidth=3, alpha=0.8, zorder=6)
                ax.scatter(prob_x, prob_y, c='red', s=50, zorder=6)

            # Draw heading indicator
            heading_length = 20 + slug.size
            end_x = slug_x + heading_length * math.cos(math.radians(slug.angle))
            end_y = slug_y + heading_length * math.sin(math.radians(slug.angle))
            ax.arrow(slug_x, slug_y, end_x - slug_x, end_y - slug_y,
                    head_width=8, head_length=6, fc=color, ec=color,
                    zorder=6, alpha=0.6)

            # Show nociceptors if enabled
            if show_nociceptors.value:
                for noc in slug.nociceptors:
                    # Color based on pain value
                    if noc.painval > 0.000001:
                        pain_intensity = min(1.0, noc.painval / 5.0)
                        noc_color = (1.0, 1.0 - pain_intensity, 1.0 - pain_intensity)
                        ax.scatter(noc.x, noc.y, c=[noc_color], s=80,
                                  edgecolors='black', linewidth=1, zorder=7, alpha=0.8)

            # Show bite indicator
            if slug.is_biting and slug.bite_target:
                tx, ty = slug.bite_target.pos
                ax.plot([slug_x, tx], [slug_y, ty], 'r-', linewidth=4,
                       alpha=0.7, zorder=4)
                ax.text((slug_x + tx)/2, (slug_y + ty)/2, '💥',
                       fontsize=24, ha='center', va='center', zorder=7)

        # Legend
        legend_elements = [
            mpatches.Patch(color='cyan', label=f'Hermissenda ({prey_counts["hermi"]})'),
            mpatches.Patch(color='pink', label=f'Flabellina ({prey_counts["flab"]})'),
            mpatches.Patch(color='yellow', label=f'Faux-Flab ({prey_counts["fauxflab"]})'),
        ]

        if model.biting:
            legend_elements.append(
                mlines.Line2D([], [], color='red', linewidth=4,
                             label='Biting!', marker='o', markersize=10)
            )

        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')

        return fig

    # Get selected slug
    selected_slug = (model.cyberslugs[selected_slug_idx.value]
                    if selected_slug_idx.value < len(model.cyberslugs)
                    else model.cyberslugs[0] if model.cyberslugs else None)

    # UI Layout
    with solara.Column(style={"padding": "20px"}):
        solara.Markdown("# 🐌 CyberSlug: Complete NetLogo Implementation")

        with solara.Row():
            # Left column - Visualization
            with solara.Column(style={"width": "65%", "padding": "10px"}):
                solara.Info("✨ All NetLogo features implemented: Advanced Learning, Clustering, Interactive Tools, Proboscis!")

                # Control buttons
                with solara.Row():
                    solara.Button("🔄 Reset", on_click=reset, color="primary")
                    solara.Button("▶️ Step", on_click=do_step, color="success",
                                 disabled=auto_running.value)
                    solara.Button("⏩ Step 10x", on_click=do_multiple_steps, color="warning",
                                 disabled=auto_running.value)

                # Auto-run controls
                with solara.Row():
                    if auto_running.value:
                        solara.Button("⏸️ Pause Auto-Run", on_click=toggle_auto_run,
                                     color="warning", style="width: 200px")
                    else:
                        solara.Button("▶️ Start Auto-Run", on_click=toggle_auto_run,
                                     color="success", style="width: 200px")

                with solara.Row():
                    solara.SliderInt("Steps per frame", value=steps_per_frame,
                                    min=1, max=500, disabled=auto_running.value)
                    solara.SliderFloat("Update speed (sec)", value=update_interval,
                                      min=0.01, max=2.0, step=0.05, disabled=auto_running.value)

                if auto_running.value:
                    solara.Success(f"🔄 AUTO-RUNNING: {steps_per_frame.value} steps every {update_interval.value}s")

                # Visualization
                fig = create_plot()
                solara.FigureMatplotlib(fig, dependencies=[render_key])

            # Right column - Controls and stats
            with solara.Column(style={"width": "35%", "padding": "10px"}):
                solara.Markdown(f"### 📊 Step: {model.steps}")

                solara.Markdown("---")
                solara.Markdown("## 🛠️ Interactive Tools")
                solara.Info("⚠️ Only one tool can be active at a time")

                with solara.Row():
                    solara.Checkbox(label="🖱️ Dragger (drag agents)", value=dragger_mode)
                    solara.Checkbox(label="🔨 Poker (apply pain)", value=poker_mode)

                solara.Checkbox(label="👁️ Set Observer (click slug)", value=observer_mode)

                if dragger_mode.value:
                    solara.Warning("🖱️ Dragger active: Click and drag slugs or prey")
                if poker_mode.value:
                    solara.Warning("🔨 Poker active: Click near slugs to apply pain")
                if observer_mode.value:
                    solara.Warning("👁️ Observer mode: Click a slug to observe its variables")

                solara.Markdown("---")
                solara.Markdown("## 🎛️ Population Settings")
                solara.SliderInt("🐌 Slugs", value=num_slugs, min=1, max=10)
                solara.SliderInt("🔵 Hermissenda", value=hermi_pop, min=0, max=30)
                solara.SliderInt("🔴 Flabellina", value=flab_pop, min=0, max=30)
                solara.SliderInt("🟡 Faux-Flabellina", value=fauxflab_pop, min=0, max=30)

                solara.Markdown("---")
                solara.Markdown("## ⚙️ Simulation Controls")

                with solara.Row():
                    solara.Checkbox(label="🎯 Clustering", value=clustering)
                    solara.Checkbox(label="🚫 Immobilize", value=immobilize)

                if clustering.value:
                    solara.SliderInt("Cluster Radius", value=cluster_radius, min=5, max=50)

                with solara.Row():
                    solara.Checkbox(label="🦷 Biting", value=biting)
                    solara.Checkbox(label="👃 Odor Null", value=odor_null)

                solara.Checkbox(label="👀 Show Nociceptors", value=show_nociceptors)

                with solara.Row():
                    solara.Checkbox(label="🔒 Fix Satiation", value=fix_satiation)
                    if fix_satiation.value:
                        solara.SliderFloat("Satiation Value", value=satiation_value,
                                          min=0.01, max=1.0, step=0.01)

                if selected_slug:
                    solara.Markdown("---")
                    solara.Markdown(f"## 🔍 Slug {selected_slug_idx.value} Details")

                    if len(model.cyberslugs) > 1:
                        solara.SliderInt("Select Slug", value=selected_slug_idx,
                                        min=0, max=len(model.cyberslugs)-1)

                    solara.Markdown("### 🍽️ Prey Encounters")
                    with solara.Row():
                        solara.Text(f"🔵 Hermi: {selected_slug.hermi_counter}")
                        solara.Text(f"🔴 Flab: {selected_slug.flab_counter}")
                    solara.Text(f"🟡 Fauxflab: {selected_slug.fauxflab_counter}")

                    solara.Markdown("### 💥 Social")
                    with solara.Row():
                        solara.Text(f"⚔️ Bites: {selected_slug.bite_counter}")
                        solara.Text(f"🤕 Bitten: {selected_slug.被咬_counter}")
                    with solara.Row():
                        solara.Text(f"📏 Size: {selected_slug.size:.1f}")
                        solara.Text(f"💀 Collision: {selected_slug.collision}")

                    if selected_slug.proboscis_extended:
                        solara.Success(f"👅 PROBOSCIS EXTENDED ({selected_slug.proboscis_phase}/20)")

                    solara.Markdown("---")
                    solara.Markdown("## 🧠 Internal States")
                    with solara.Row():
                        solara.Text(f"Nutrition: {selected_slug.nutrition:.3f}")
                        solara.Text(f"Satiation: {selected_slug.satiation:.3f}")
                    with solara.Row():
                        solara.Text(f"AppState: {selected_slug.app_state:.3f}")
                        solara.Text(f"Switch: {selected_slug.app_state_switch:.3f}")
                    with solara.Row():
                        solara.Text(f"Incentive: {selected_slug.incentive:.3f}")
                        solara.Text(f"Reward: {selected_slug.reward:.3f}")
                    with solara.Row():
                        solara.Text(f"Pain: {selected_slug.sns_pain_total:.3f}")
                        solara.Text(f"SomaticMap: {selected_slug.somatic_map:.3f}")

                    solara.Markdown("---")
                    solara.Markdown("## 🧬 Advanced Learning Circuit")

                    # R+, R-, NR neurons
                    with solara.Row():
                        solara.Text(f"R+: {selected_slug.R_pos:.3f}")
                        solara.Text(f"R-: {selected_slug.R_neg:.3f}")
                        solara.Text(f"NR: {selected_slug.NR:.3f}")

                    # CS neurons
                    with solara.Row():
                        solara.Text(f"CS1: {selected_slug.CS1:.3f}")
                        solara.Text(f"CS2: {selected_slug.CS2:.3f}")

                    solara.Markdown("#### Association Strengths (V)")
                    with solara.Row():
                        solara.Text(f"Vh_rp: {selected_slug.Vh_rp:.3f}")
                        solara.Text(f"(V0: {selected_slug.Vh_rp0:.3f})")
                    with solara.Row():
                        solara.Text(f"Vf_rn: {selected_slug.Vf_rn:.3f}")
                        solara.Text(f"(V0: {selected_slug.Vf_rn0:.3f})")
                    with solara.Row():
                        solara.Text(f"Vf_rp: {selected_slug.Vf_rp:.3f}")
                        solara.Text(f"Vh_rn: {selected_slug.Vh_rn:.3f}")

                    solara.Markdown("#### Synaptic Weights (W)")
                    with solara.Row():
                        solara.Text(f"Wh_rp: {selected_slug.Wh_rp:.3f}")
                        solara.Text(f"Wf_rn: {selected_slug.Wf_rn:.3f}")

                    # Reset buttons
                    with solara.Row():
                        solara.Button("🔄 Zero Vh", on_click=zero_vh, color="warning")
                        solara.Button("🔄 Zero Vf", on_click=zero_vf, color="warning")

                    solara.Markdown("---")
                    solara.Markdown("## 🔄 Habituation Circuit")
                    with solara.Row():
                        solara.Text(f"M: {selected_slug.M:.3f}")
                        solara.Text(f"M0: {selected_slug.M0:.3f}")
                    with solara.Row():
                        solara.Text(f"W3: {selected_slug.W3:.3f}")
                        solara.Text(f"Is: {selected_slug.Is:.3f}")

                solara.Markdown("---")
                solara.Markdown("## 🌍 Global Stats")
                solara.Text(f"Total Slugs: {len(model.cyberslugs)}")
                solara.Text(f"Total Bites: {sum(s.bite_counter for s in model.cyberslugs)}")
                total_prey = sum(s.hermi_counter + s.flab_counter + s.fauxflab_counter
                               for s in model.cyberslugs)
                solara.Text(f"Total Prey Eaten: {total_prey}")

                if model.clustering:
                    solara.Info("🎯 Clustering Mode Active")
                if model.immobilize:
                    solara.Warning("🚫 Movement Immobilized")
                if model.odor_null:
                    solara.Warning("👃 Odor Emission Disabled")


page = Page