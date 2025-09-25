import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
from model import CyberslugModel


class CyberslugGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🐌 Cyberslug Neural Simulation Interface")
        self.root.geometry("1400x900")  # Made wider to fit more data

        self.model = None
        self.running = False
        self.step_count = 0

        self.setup_interface()

    def setup_interface(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Left Panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="🎛️ Simulation Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        # Population Controls
        ttk.Label(control_frame, text="🎯 Prey Populations", font=("Arial", 12, "bold")).grid(row=0, column=0,
                                                                                             columnspan=2, pady=(0, 10))

        ttk.Label(control_frame, text="Hermi:").grid(row=1, column=0, sticky=tk.W)
        self.hermi_var = tk.IntVar(value=4)
        ttk.Scale(control_frame, from_=0, to=20, variable=self.hermi_var, orient=tk.HORIZONTAL).grid(row=1, column=1,
                                                                                                     sticky=(tk.W,
                                                                                                             tk.E))
        self.hermi_label = ttk.Label(control_frame, text="4")
        self.hermi_label.grid(row=1, column=2)

        ttk.Label(control_frame, text="Flab:").grid(row=2, column=0, sticky=tk.W)
        self.flab_var = tk.IntVar(value=4)
        ttk.Scale(control_frame, from_=0, to=20, variable=self.flab_var, orient=tk.HORIZONTAL).grid(row=2, column=1,
                                                                                                    sticky=(tk.W, tk.E))
        self.flab_label = ttk.Label(control_frame, text="4")
        self.flab_label.grid(row=2, column=2)

        ttk.Label(control_frame, text="FauxFlab:").grid(row=3, column=0, sticky=tk.W)
        self.fauxflab_var = tk.IntVar(value=4)
        ttk.Scale(control_frame, from_=0, to=20, variable=self.fauxflab_var, orient=tk.HORIZONTAL).grid(row=3, column=1,
                                                                                                        sticky=(tk.W,
                                                                                                                tk.E))
        self.fauxflab_label = ttk.Label(control_frame, text="4")
        self.fauxflab_label.grid(row=3, column=2)

        # Learning Rate Controls
        ttk.Label(control_frame, text="🧠 Learning Rates", font=("Arial", 12, "bold")).grid(row=4, column=0,
                                                                                           columnspan=2, pady=(20, 10))

        ttk.Label(control_frame, text="Hermi Alpha:").grid(row=5, column=0, sticky=tk.W)
        self.alpha_hermi_var = tk.DoubleVar(value=0.5)
        ttk.Scale(control_frame, from_=0.0, to=1.0, variable=self.alpha_hermi_var, orient=tk.HORIZONTAL).grid(row=5,
                                                                                                              column=1,
                                                                                                              sticky=(
                                                                                                                  tk.W,
                                                                                                                  tk.E))
        self.alpha_hermi_label = ttk.Label(control_frame, text="0.5")
        self.alpha_hermi_label.grid(row=5, column=2)

        ttk.Label(control_frame, text="Flab Alpha:").grid(row=6, column=0, sticky=tk.W)
        self.alpha_flab_var = tk.DoubleVar(value=0.5)
        ttk.Scale(control_frame, from_=0.0, to=1.0, variable=self.alpha_flab_var, orient=tk.HORIZONTAL).grid(row=6,
                                                                                                             column=1,
                                                                                                             sticky=(
                                                                                                                 tk.W,
                                                                                                                 tk.E))
        self.alpha_flab_label = ttk.Label(control_frame, text="0.5")
        self.alpha_flab_label.grid(row=6, column=2)

        # Simulation Speed
        ttk.Label(control_frame, text="⏱️ Speed", font=("Arial", 12, "bold")).grid(row=7, column=0, columnspan=2,
                                                                                   pady=(20, 10))
        ttk.Label(control_frame, text="Steps/sec:").grid(row=8, column=0, sticky=tk.W)
        self.speed_var = tk.DoubleVar(value=2.0)
        ttk.Scale(control_frame, from_=0.1, to=10.0, variable=self.speed_var, orient=tk.HORIZONTAL).grid(row=8,
                                                                                                         column=1,
                                                                                                         sticky=(tk.W,
                                                                                                                 tk.E))
        self.speed_label = ttk.Label(control_frame, text="2.0")
        self.speed_label.grid(row=8, column=2)

        # Control Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=9, column=0, columnspan=3, pady=(20, 0), sticky=(tk.W, tk.E))

        self.start_button = ttk.Button(button_frame, text="🚀 Start Simulation", command=self.start_simulation)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))

        self.stop_button = ttk.Button(button_frame, text="⏹️ Stop", command=self.stop_simulation, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 5))

        self.reset_button = ttk.Button(button_frame, text="🔄 Reset", command=self.reset_simulation)
        self.reset_button.pack(side=tk.LEFT)

        # Middle Panel - Neural State Data
        neural_frame = ttk.LabelFrame(main_frame, text="🧠 Neural State", padding="10")
        neural_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        # Basic Status
        basic_frame = ttk.LabelFrame(neural_frame, text="📊 Basic Status", padding="5")
        basic_frame.pack(fill=tk.X, pady=(0, 10))

        self.step_label = ttk.Label(basic_frame, text="Step: 0", font=("Arial", 10, "bold"))
        self.step_label.pack(anchor=tk.W)

        self.nutrition_label = ttk.Label(basic_frame, text="Nutrition: 0.500")
        self.nutrition_label.pack(anchor=tk.W)

        self.app_state_label = ttk.Label(basic_frame, text="Appetitive State: 0.500")
        self.app_state_label.pack(anchor=tk.W)

        self.incentive_label = ttk.Label(basic_frame, text="Incentive: 0.000")
        self.incentive_label.pack(anchor=tk.W)

        # Advanced Neural Variables
        advanced_frame = ttk.LabelFrame(neural_frame, text="⚗️ Advanced Neural Variables", padding="5")
        advanced_frame.pack(fill=tk.X, pady=(0, 10))

        self.satiation_label = ttk.Label(advanced_frame, text="Satiation: 0.000")
        self.satiation_label.pack(anchor=tk.W)

        self.pain_label = ttk.Label(advanced_frame, text="Pain: 0.000")
        self.pain_label.pack(anchor=tk.W)

        self.somatic_map_label = ttk.Label(advanced_frame, text="Somatic Map: 0.000")
        self.somatic_map_label.pack(anchor=tk.W)

        self.app_state_switch_label = ttk.Label(advanced_frame, text="App State Switch: 0.000")
        self.app_state_switch_label.pack(anchor=tk.W)

        # Reward System
        reward_frame = ttk.LabelFrame(neural_frame, text="🎁 Reward System", padding="5")
        reward_frame.pack(fill=tk.X, pady=(0, 10))

        self.reward_pos_label = ttk.Label(reward_frame, text="Reward Positive: 0.000")
        self.reward_pos_label.pack(anchor=tk.W)

        self.reward_neg_label = ttk.Label(reward_frame, text="Reward Negative: 0.000")
        self.reward_neg_label.pack(anchor=tk.W)

        self.turn_angle_label = ttk.Label(reward_frame, text="Turn Angle: 0.000")
        self.turn_angle_label.pack(anchor=tk.W)

        # Value Functions
        value_frame = ttk.LabelFrame(neural_frame, text="💡 Learned Values", padding="5")
        value_frame.pack(fill=tk.X)

        self.vh_label = ttk.Label(value_frame, text="Vh (Hermi Value): 0.000")
        self.vh_label.pack(anchor=tk.W)

        self.vf_label = ttk.Label(value_frame, text="Vf (Flab Value): 0.000")
        self.vf_label.pack(anchor=tk.W)

        self.vd_label = ttk.Label(value_frame, text="Vd (Drug Value): 0.000")
        self.vd_label.pack(anchor=tk.W)

        # Right Panel - Data Display and Learning Progress
        data_frame = ttk.LabelFrame(main_frame, text="📈 Learning Progress & Log", padding="10")
        data_frame.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Encounter Counts
        encounter_frame = ttk.LabelFrame(data_frame, text="🎯 Encounters", padding="5")
        encounter_frame.pack(fill=tk.X, pady=(0, 10))

        self.hermi_encounter_label = ttk.Label(encounter_frame, text="Hermi Encounters: 0")
        self.hermi_encounter_label.pack(anchor=tk.W)

        self.flab_encounter_label = ttk.Label(encounter_frame, text="Flab Encounters: 0")
        self.flab_encounter_label.pack(anchor=tk.W)

        self.drug_encounter_label = ttk.Label(encounter_frame, text="Drug Encounters: 0")
        self.drug_encounter_label.pack(anchor=tk.W)

        # Sensory Input Display
        sensory_frame = ttk.LabelFrame(data_frame, text="👃 Sensory Input", padding="5")
        sensory_frame.pack(fill=tk.X, pady=(0, 10))

        self.sns_betaine_label = ttk.Label(sensory_frame, text="Betaine: 0.000")
        self.sns_betaine_label.pack(anchor=tk.W)

        self.sns_hermi_label = ttk.Label(sensory_frame, text="Hermi Odor: 0.000")
        self.sns_hermi_label.pack(anchor=tk.W)

        self.sns_flab_label = ttk.Label(sensory_frame, text="Flab Odor: 0.000")
        self.sns_flab_label.pack(anchor=tk.W)

        self.sns_drug_label = ttk.Label(sensory_frame, text="Drug Odor: 0.000")
        self.sns_drug_label.pack(anchor=tk.W)

        # Live Log
        log_frame = ttk.LabelFrame(data_frame, text="📝 Live Log", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=12, width=40)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Update slider labels
        self.update_slider_labels()

        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(0, weight=1)
        data_frame.rowconfigure(3, weight=1)

    def update_slider_labels(self):
        """Update slider value labels"""
        self.hermi_label.config(text=str(self.hermi_var.get()))
        self.flab_label.config(text=str(self.flab_var.get()))
        self.fauxflab_label.config(text=str(self.fauxflab_var.get()))
        self.alpha_hermi_label.config(text=f"{self.alpha_hermi_var.get():.2f}")
        self.alpha_flab_label.config(text=f"{self.alpha_flab_var.get():.2f}")
        self.speed_label.config(text=f"{self.speed_var.get():.1f}")

        # Schedule next update
        self.root.after(100, self.update_slider_labels)

    def start_simulation(self):
        """Start the simulation"""
        if not self.running:
            # Create new model with current parameters
            self.model = CyberslugModel(
                hermi_count=self.hermi_var.get(),
                flab_count=self.flab_var.get(),
                fauxflab_count=self.fauxflab_var.get()
            )

            # Update learning rates
            cyberslug = self.model.get_cyberslug()
            if cyberslug:
                cyberslug.alpha_hermi = self.alpha_hermi_var.get()
                cyberslug.alpha_flab = self.alpha_flab_var.get()

            self.running = True
            self.step_count = 0

            # Update button states
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)

            # Start simulation thread
            self.sim_thread = threading.Thread(target=self.run_simulation, daemon=True)
            self.sim_thread.start()

            self.log_message("🚀 Simulation started!")

    def stop_simulation(self):
        """Stop the simulation"""
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.log_message("⏹️ Simulation stopped.")

    def reset_simulation(self):
        """Reset the simulation"""
        self.stop_simulation()
        self.step_count = 0
        self.model = None

        # Reset basic displays
        self.step_label.config(text="Step: 0")
        self.nutrition_label.config(text="Nutrition: 0.500")
        self.app_state_label.config(text="Appetitive State: 0.500")
        self.incentive_label.config(text="Incentive: 0.000")

        # Reset advanced neural displays
        self.satiation_label.config(text="Satiation: 0.000")
        self.pain_label.config(text="Pain: 0.000")
        self.somatic_map_label.config(text="Somatic Map: 0.000")
        self.app_state_switch_label.config(text="App State Switch: 0.000")

        # Reset reward system displays
        self.reward_pos_label.config(text="Reward Positive: 0.000")
        self.reward_neg_label.config(text="Reward Negative: 0.000")
        self.turn_angle_label.config(text="Turn Angle: 0.000")

        # Reset value function displays
        self.vh_label.config(text="Vh (Hermi Value): 0.000")
        self.vf_label.config(text="Vf (Flab Value): 0.000")
        self.vd_label.config(text="Vd (Drug Value): 0.000")

        # Reset encounter displays
        self.hermi_encounter_label.config(text="Hermi Encounters: 0")
        self.flab_encounter_label.config(text="Flab Encounters: 0")
        self.drug_encounter_label.config(text="Drug Encounters: 0")

        # Reset sensory displays
        self.sns_betaine_label.config(text="Betaine: 0.000")
        self.sns_hermi_label.config(text="Hermi Odor: 0.000")
        self.sns_flab_label.config(text="Flab Odor: 0.000")
        self.sns_drug_label.config(text="Drug Odor: 0.000")

        self.log_text.delete(1.0, tk.END)
        self.log_message("🔄 Simulation reset.")

    def run_simulation(self):
        """Main simulation loop"""
        while self.running and self.model:
            try:
                # Step the model
                self.model.step()
                self.step_count += 1

                # Update display (thread-safe)
                self.root.after(0, self.update_display)

                # Control simulation speed
                time.sleep(1.0 / self.speed_var.get())

            except Exception as e:
                self.root.after(0, lambda: self.log_message(f"❌ Error: {str(e)}"))
                break

    def update_display(self):
        """Update the GUI display with current data"""
        if not self.model:
            return

        cyberslug = self.model.get_cyberslug()
        if not cyberslug:
            return

        # Update basic status labels
        self.step_label.config(text=f"Step: {self.step_count}")
        self.nutrition_label.config(text=f"Nutrition: {cyberslug.nutrition:.3f}")
        self.app_state_label.config(text=f"Appetitive State: {cyberslug.app_state:.3f}")
        self.incentive_label.config(text=f"Incentive: {cyberslug.incentive:.3f}")

        # Update advanced neural variables
        self.satiation_label.config(text=f"Satiation: {cyberslug.satiation:.3f}")
        self.pain_label.config(text=f"Pain: {cyberslug.pain:.3f}")
        self.somatic_map_label.config(text=f"Somatic Map: {cyberslug.somatic_map:.3f}")
        self.app_state_switch_label.config(text=f"App State Switch: {cyberslug.app_state_switch:.3f}")

        # Update reward system
        self.reward_pos_label.config(text=f"Reward Positive: {cyberslug.reward_pos:.3f}")
        self.reward_neg_label.config(text=f"Reward Negative: {cyberslug.reward_neg:.3f}")
        self.turn_angle_label.config(text=f"Turn Angle: {cyberslug.turn_angle:.3f}")

        # Update value functions (learning)
        self.vh_label.config(text=f"Vh (Hermi Value): {cyberslug.Vh:.3f}")
        self.vf_label.config(text=f"Vf (Flab Value): {cyberslug.Vf:.3f}")
        self.vd_label.config(text=f"Vd (Drug Value): {cyberslug.Vd:.3f}")

        # Update encounter counts
        self.hermi_encounter_label.config(text=f"Hermi Encounters: {cyberslug.hermi_counter}")
        self.flab_encounter_label.config(text=f"Flab Encounters: {cyberslug.flab_counter}")
        self.drug_encounter_label.config(text=f"Drug Encounters: {cyberslug.drug_counter}")

        # Update sensory input
        if len(cyberslug.sns_odors) >= 4:
            self.sns_betaine_label.config(text=f"Betaine: {cyberslug.sns_odors[0]:.3f}")
            self.sns_hermi_label.config(text=f"Hermi Odor: {cyberslug.sns_odors[1]:.3f}")
            self.sns_flab_label.config(text=f"Flab Odor: {cyberslug.sns_odors[2]:.3f}")
            self.sns_drug_label.config(text=f"Drug Odor: {cyberslug.sns_odors[3]:.3f}")

        # Log important events and state changes
        if self.step_count % 20 == 0:
            behavior = "APPROACH" if cyberslug.app_state > 0.5 else "AVOIDANCE"
            self.log_message(f"Step {self.step_count}: {behavior} mode")
            self.log_message(f"  Nutrition={cyberslug.nutrition:.3f}, Satiation={cyberslug.satiation:.3f}")

        # Log encounters
        total_encounters = cyberslug.hermi_counter + cyberslug.flab_counter + cyberslug.drug_counter
        if hasattr(self, 'last_encounter_total'):
            if total_encounters > self.last_encounter_total:
                encounter_types = []
                if cyberslug.hermi_counter > getattr(self, 'last_hermi', 0):
                    encounter_types.append("HERMI")
                    self.last_hermi = cyberslug.hermi_counter
                if cyberslug.flab_counter > getattr(self, 'last_flab', 0):
                    encounter_types.append("FLAB")
                    self.last_flab = cyberslug.flab_counter
                if cyberslug.drug_counter > getattr(self, 'last_drug', 0):
                    encounter_types.append("DRUG")
                    self.last_drug = cyberslug.drug_counter

                if encounter_types:
                    self.log_message(f"🎯 {' + '.join(encounter_types)} encounter! Total: {total_encounters}")

        self.last_encounter_total = total_encounters

    def log_message(self, message):
        """Add message to log"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

    def run(self):
        """Start the GUI"""
        self.log_message("🐌 Cyberslug Neural Simulation Interface Ready")
        self.log_message("📋 Adjust parameters and click 'Start Simulation'")
        self.log_message("🧠 This interface shows complete neural state information")
        self.last_encounter_total = 0  # Initialize encounter tracking
        self.root.mainloop()


if __name__ == "__main__":
    app = CyberslugGUI()
    app.run()
