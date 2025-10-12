"""
run.py - Enhanced execution script for CyberSlug simulation
Now with ALL NetLogo features including advanced learning circuit
"""
from model import CyberSlugModel
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def run_simulation(steps=1000,
                   num_slugs=1,
                   hermi=15,
                   flab=15,
                   fauxflab=15,
                   clustering=False,
                   immobilize=False):
    """Run a single simulation with all features"""
    model = CyberSlugModel(
        num_slugs=num_slugs,
        hermi_population=hermi,
        flab_population=flab,
        fauxflab_population=fauxflab,
        clustering=clustering,
        immobilize=immobilize
    )

    print(f"Running simulation for {steps} steps...")
    print(f"Slugs: {num_slugs}")
    print(f"Populations - Hermi: {hermi}, Flab: {flab}, FauxFlab: {fauxflab}")
    print(f"Clustering: {clustering}, Immobilize: {immobilize}")

    for i in range(steps):
        model.step()
        if (i + 1) % 100 == 0:
            print(f"Step {i + 1}/{steps} completed")

    print("\nSimulation complete!")
    return model


def plot_results(model):
    """Plot comprehensive simulation results"""
    data = model.datacollector.get_model_vars_dataframe()

    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('CyberSlug Complete Simulation Results', fontsize=16, fontweight='bold')

    # Plot 1: Prey encounters over time
    axes[0, 0].plot(data.index, data['Total_Hermi_Eaten'], label='Hermissenda',
                   color='cyan', linewidth=2)
    axes[0, 0].plot(data.index, data['Total_Flab_Eaten'], label='Flabellina',
                   color='pink', linewidth=2)
    axes[0, 0].plot(data.index, data['Total_Fauxflab_Eaten'], label='Faux-Flab',
                   color='yellow', linewidth=2)
    axes[0, 0].set_xlabel('Time Steps')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Prey Encounters')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Appetitive state over time
    axes[0, 1].plot(data.index, data['Avg_AppState'], color='blue', linewidth=2)
    axes[0, 1].set_xlabel('Time Steps')
    axes[0, 1].set_ylabel('Appetitive State')
    axes[0, 1].set_title('Average Appetitive State')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Learning values - Vh_rp (Hermi positive reward)
    axes[1, 0].plot(data.index, data['Avg_Vh_rp'], color='green', linewidth=2)
    axes[1, 0].set_xlabel('Time Steps')
    axes[1, 0].set_ylabel('Association Strength')
    axes[1, 0].set_title('Vh_rp (Hermi → Reward+)')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Learning values - Vf_rn (Flab negative reward)
    axes[1, 1].plot(data.index, data['Avg_Vf_rn'], color='red', linewidth=2)
    axes[1, 1].set_xlabel('Time Steps')
    axes[1, 1].set_ylabel('Association Strength')
    axes[1, 1].set_title('Vf_rn (Flab → Reward-)')
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 5: Nutrition over time
    axes[2, 0].plot(data.index, data['Avg_Nutrition'], color='orange', linewidth=2)
    axes[2, 0].set_xlabel('Time Steps')
    axes[2, 0].set_ylabel('Nutrition Level')
    axes[2, 0].set_title('Average Nutrition')
    axes[2, 0].grid(True, alpha=0.3)

    # Plot 6: Social interactions (bites)
    axes[2, 1].plot(data.index, data['Total_Bites'], color='red', linewidth=2)
    axes[2, 1].set_xlabel('Time Steps')
    axes[2, 1].set_ylabel('Bite Count')
    axes[2, 1].set_title('Total Bites (Social Interactions)')
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cyberslug_complete_results.png', dpi=300, bbox_inches='tight')
    print("\nPlots saved to 'cyberslug_complete_results.png'")
    plt.show()

    return data


def print_summary(model):
    """Print comprehensive summary statistics"""
    print("\n" + "=" * 70)
    print("CYBERSLUG SIMULATION SUMMARY")
    print("=" * 70)
    print(f"Total Steps: {model.ticks}")
    print(f"Number of Slugs: {len(model.cyberslugs)}")

    print("\n" + "-" * 70)
    print("GLOBAL STATISTICS")
    print("-" * 70)

    total_hermi = sum(s.hermi_counter for s in model.cyberslugs)
    total_flab = sum(s.flab_counter for s in model.cyberslugs)
    total_fauxflab = sum(s.fauxflab_counter for s in model.cyberslugs)
    total_bites = sum(s.bite_counter for s in model.cyberslugs)

    print(f"Total Hermissenda eaten: {total_hermi}")
    print(f"Total Flabellina eaten: {total_flab}")
    print(f"Total Faux-Flabellina eaten: {total_fauxflab}")
    print(f"Total Bites: {total_bites}")

    avg_nutrition = np.mean([s.nutrition for s in model.cyberslugs])
    avg_size = np.mean([s.size for s in model.cyberslugs])

    print(f"\nAverage Nutrition: {avg_nutrition:.3f}")
    print(f"Average Size: {avg_size:.3f}")

    print("\n" + "-" * 70)
    print("PER-SLUG DETAILS")
    print("-" * 70)

    for i, slug in enumerate(model.cyberslugs):
        print(f"\n🐌 Slug {i}:")
        print(f"  Size: {slug.size:.2f}")
        print(f"  Nutrition: {slug.nutrition:.3f}")
        print(f"  Satiation: {slug.satiation:.3f}")
        print(f"  Appetitive State: {slug.app_state:.3f}")

        print(f"\n  Prey Consumed:")
        print(f"    Hermissenda: {slug.hermi_counter}")
        print(f"    Flabellina: {slug.flab_counter}")
        print(f"    Faux-Flabellina: {slug.fauxflab_counter}")

        print(f"\n  Social Interactions:")
        print(f"    Bites Given: {slug.bite_counter}")
        print(f"    Times Bitten: {slug.被咬_counter}")

        print(f"\n  Learning Circuit (Association Strengths):")
        print(f"    Vh_rp (Hermi→R+): {slug.Vh_rp:.3f} (baseline: {slug.Vh_rp0:.3f})")
        print(f"    Vh_rn (Hermi→R-): {slug.Vh_rn:.3f} (baseline: {slug.Vh_rn0:.3f})")
        print(f"    Vf_rp (Flab→R+):  {slug.Vf_rp:.3f} (baseline: {slug.Vf_rp0:.3f})")
        print(f"    Vf_rn (Flab→R-):  {slug.Vf_rn:.3f} (baseline: {slug.Vf_rn0:.3f})")
        print(f"    Vh_n (Hermi→NR):  {slug.Vh_n:.3f}")
        print(f"    Vf_n (Flab→NR):   {slug.Vf_n:.3f}")

        print(f"\n  Synaptic Weights:")
        print(f"    Wh_rp: {slug.Wh_rp:.3f}")
        print(f"    Wf_rn: {slug.Wf_rn:.3f}")

        print(f"\n  Reward Neurons:")
        print(f"    R+: {slug.R_pos:.3f}")
        print(f"    R-: {slug.R_neg:.3f}")
        print(f"    NR: {slug.NR:.3f}")

        print(f"\n  Habituation Circuit:")
        print(f"    M (processed odor): {slug.M:.3f}")
        print(f"    M0 (baseline): {slug.M0:.3f}")
        print(f"    W3 (synaptic weight): {slug.W3:.3f}")

    print("\n" + "=" * 70)


def run_learning_experiment():
    """
    Run a learning experiment to demonstrate the advanced circuit
    """
    print("\n" + "=" * 70)
    print("LEARNING EXPERIMENT: Training slug on Hermissenda")
    print("=" * 70)

    model = CyberSlugModel(
        num_slugs=1,
        hermi_population=20,
        flab_population=5,
        fauxflab_population=0
    )

    slug = model.cyberslugs[0]

    # Track learning over time
    steps_to_track = [0, 100, 500, 1000, 2000]
    learning_data = []

    for step in range(max(steps_to_track) + 1):
        model.step()

        if step in steps_to_track:
            learning_data.append({
                'step': step,
                'Vh_rp': slug.Vh_rp,
                'Vh_rp0': slug.Vh_rp0,
                'Wh_rp': slug.Wh_rp,
                'hermi_eaten': slug.hermi_counter,
                'R_pos': slug.R_pos,
                'CS1': slug.CS1
            })

            print(f"\nStep {step}:")
            print(f"  Hermissenda eaten: {slug.hermi_counter}")
            print(f"  Vh_rp: {slug.Vh_rp:.4f} (baseline: {slug.Vh_rp0:.4f})")
            print(f"  Wh_rp: {slug.Wh_rp:.4f}")
            print(f"  R+ activity: {slug.R_pos:.4f}")
            print(f"  CS1 (hermi trace): {slug.CS1:.4f}")

    print("\n" + "=" * 70)
    print("Learning complete! Slug should show increased Vh_rp and Wh_rp values.")
    print("=" * 70)

    return learning_data


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--learning":
        # Run learning experiment
        learning_data = run_learning_experiment()
    else:
        # Run standard simulation
        model = run_simulation(
            steps=2000,
            num_slugs=2,
            hermi=15,
            flab=15,
            fauxflab=15,
            clustering=False
        )

        # Print summary
        print_summary(model)

        # Plot results
        data = plot_results(model)

        # Save data to CSV
        data.to_csv('cyberslug_complete_data.csv')
        print("\nData saved to 'cyberslug_complete_data.csv'")

        print("\n💡 TIP: Run with '--learning' flag to see learning experiment:")
        print("   python run.py --learning")