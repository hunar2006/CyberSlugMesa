from model import CyberslugModel
import time


def run_interactive_simulation():
    """Interactive Cyberslug simulation with parameter control"""

    print("🐌 Interactive Cyberslug Neural Simulation")
    print("=" * 50)

    while True:
        print("\n🔧 PARAMETER SETUP")
        print("1. Set Prey Populations")
        print("2. Set Learning Rates")
        print("3. Set Simulation Length")
        print("4. Quick Start (default settings)")
        print("5. Exit")

        choice = input("\nChoose option (1-5): ").strip()

        if choice == "5":
            break

        # Default parameters
        hermi_count = 4
        flab_count = 4
        fauxflab_count = 4
        steps = 50
        alpha_hermi = 0.5
        beta_hermi = 1.0
        alpha_flab = 0.5
        beta_flab = 1.0

        if choice == "1":
            print("\n🎯 Set Prey Populations")
            hermi_count = int(input(f"Hermi population (current: {hermi_count}): ") or hermi_count)
            flab_count = int(input(f"Flab population (current: {flab_count}): ") or flab_count)
            fauxflab_count = int(input(f"FauxFlab population (current: {fauxflab_count}): ") or fauxflab_count)

        elif choice == "2":
            print("\n🧠 Set Learning Rates")
            print("Alpha = positive learning rate, Beta = negative learning rate")
            alpha_hermi = float(input(f"Hermi Alpha (current: {alpha_hermi}): ") or alpha_hermi)
            beta_hermi = float(input(f"Hermi Beta (current: {beta_hermi}): ") or beta_hermi)
            alpha_flab = float(input(f"Flab Alpha (current: {alpha_flab}): ") or alpha_flab)
            beta_flab = float(input(f"Flab Beta (current: {beta_flab}): ") or beta_flab)

        elif choice == "3":
            print("\n⏱️ Set Simulation Length")
            steps = int(input(f"Number of steps (current: {steps}): ") or steps)

        # Create model with custom parameters
        model = CyberslugModel(hermi_count=hermi_count, flab_count=flab_count, fauxflab_count=fauxflab_count)

        # Update learning rates if changed
        slug = model.get_cyberslug()
        if choice == "2":
            slug.alpha_hermi = alpha_hermi
            slug.beta_hermi = beta_hermi
            slug.alpha_flab = alpha_flab
            slug.beta_flab = beta_flab

        # Run simulation
        print(f"\n🚀 Running simulation with:")
        print(f"   Populations: H={hermi_count}, F={flab_count}, D={fauxflab_count}")
        print(f"   Learning rates: αH={slug.alpha_hermi}, βH={slug.beta_hermi}")
        print(f"   Steps: {steps}")
        print("\n📊 Results:")
        print("-" * 70)

        for step in range(steps):
            model.step()

            if step % 10 == 0 or step == steps - 1:
                print(f"Step {step:3d}: "
                      f"Nutrition={slug.nutrition:.3f} | "
                      f"AppState={slug.app_state:.3f} | "
                      f"Incentive={slug.incentive:.3f} | "
                      f"Encounters: H={slug.hermi_counter} F={slug.flab_counter} D={slug.drug_counter}")

        # Final analysis
        print(f"\n📈 FINAL ANALYSIS:")
        print(f"   Total Encounters: {slug.hermi_counter + slug.flab_counter + slug.drug_counter}")
        print(
            f"   Learning Success Rate: {(slug.hermi_counter + slug.flab_counter) / max(1, slug.hermi_counter + slug.flab_counter + slug.drug_counter) * 100:.1f}%")
        print(f"   Final Appetitive State: {'APPROACH' if slug.app_state > 0.5 else 'AVOIDANCE'}")

        # Option to save results
        save = input("\n💾 Save results to file? (y/n): ").strip().lower()
        if save == 'y':
            filename = f"cyberslug_results_H{hermi_count}_F{flab_count}_D{fauxflab_count}.txt"
            with open(filename, 'w') as f:
                f.write(f"Cyberslug Simulation Results\n")
                f.write(f"Parameters: H={hermi_count}, F={flab_count}, D={fauxflab_count}\n")
                f.write(f"Learning rates: αH={slug.alpha_hermi}, βH={slug.beta_hermi}\n")
                f.write(f"Final encounters: H={slug.hermi_counter}, F={slug.flab_counter}, D={slug.drug_counter}\n")
                f.write(f"Final nutrition: {slug.nutrition:.3f}\n")
                f.write(f"Final app state: {slug.app_state:.3f}\n")
            print(f"✅ Results saved to {filename}")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    run_interactive_simulation()
