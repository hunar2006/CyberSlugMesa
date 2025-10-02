"""
run.py - Basic execution script for CyberSlug simulation
Run this for batch simulations without visualization
"""
from model import CyberSlugModel
import matplotlib.pyplot as plt
import pandas as pd


def run_simulation(steps=1000, hermi=4, flab=4, fauxflab=4):
    """Run a single simulation"""
    model = CyberSlugModel(
        hermi_population=hermi,
        flab_population=flab,
        fauxflab_population=fauxflab
    )

    print(f"Running simulation for {steps} steps...")
    print(f"Populations - Hermi: {hermi}, Flab: {flab}, FauxFlab: {fauxflab}")

    for i in range(steps):
        model.step()
        if (i + 1) % 100 == 0:
            print(f"Step {i + 1}/{steps} completed")

    print("\nSimulation complete!")
    return model


def plot_results(model):
    """Plot simulation results"""
    # Get data from datacollector
    data = model.datacollector.get_model_vars_dataframe()

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('CyberSlug Simulation Results', fontsize=16)

    # Plot 1: Prey encounters over time
    axes[0, 0].plot(data.index, data['Hermi_Eaten'], label='Hermissenda', color='cyan')
    axes[0, 0].plot(data.index, data['Flab_Eaten'], label='Flabellina', color='pink')
    axes[0, 0].plot(data.index, data['Fauxflab_Eaten'], label='Faux-Flabellina', color='yellow')
    axes[0, 0].set_xlabel('Time Steps')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Prey Encounters')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Appetitive state over time
    axes[0, 1].plot(data.index, data['Appetitive_State'], color='blue')
    axes[0, 1].set_xlabel('Time Steps')
    axes[0, 1].set_ylabel('Appetitive State')
    axes[0, 1].set_title('Appetitive State Over Time')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Incentive over time
    axes[1, 0].plot(data.index, data['Incentive'], color='green')
    axes[1, 0].set_xlabel('Time Steps')
    axes[1, 0].set_ylabel('Incentive')
    axes[1, 0].set_title('Incentive Over Time')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Nutrition over time
    axes[1, 1].plot(data.index, data['Nutrition'], color='orange')
    axes[1, 1].set_xlabel('Time Steps')
    axes[1, 1].set_ylabel('Nutrition Level')
    axes[1, 1].set_title('Nutrition Over Time')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cyberslug_results.png', dpi=300, bbox_inches='tight')
    print("\nPlots saved to 'cyberslug_results.png'")
    plt.show()

    return data


def print_summary(model):
    """Print summary statistics"""
    print("\n" + "=" * 50)
    print("SIMULATION SUMMARY")
    print("=" * 50)
    print(f"Total Steps: {model.ticks}")
    print(f"\nPrey Encounters:")
    print(f"  Hermissenda eaten: {model.cyberslug.hermi_counter}")
    print(f"  Flabellina eaten: {model.cyberslug.flab_counter}")
    print(f"  Faux-Flabellina eaten: {model.cyberslug.fauxflab_counter}")
    print(f"\nFinal State:")
    print(f"  Nutrition: {model.cyberslug.nutrition:.3f}")
    print(f"  Satiation: {model.cyberslug.satiation:.3f}")
    print(f"  Incentive: {model.cyberslug.incentive:.3f}")
    print(f"  Appetitive State: {model.cyberslug.app_state:.3f}")
    print(f"  Somatic Map: {model.cyberslug.somatic_map:.3f}")
    print(f"\nLearning Values:")
    print(f"  V_Hermi: {model.cyberslug.Vh:.3f}")
    print(f"  V_Flab: {model.cyberslug.Vf:.3f}")
    print(f"  V_Drug: {model.cyberslug.Vd:.3f}")
    print("=" * 50)


if __name__ == "__main__":
    # Run simulation
    model = run_simulation(steps=1000, hermi=4, flab=4, fauxflab=4)

    # Print summary
    print_summary(model)

    # Plot results
    data = plot_results(model)

    # Optionally save data to CSV
    data.to_csv('cyberslug_data.csv')
    print("Data saved to 'cyberslug_data.csv'")