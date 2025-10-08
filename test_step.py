from model import CyberSlugModel

model = CyberSlugModel()
print(f"After creation, steps: {model.steps}")

model.my_step()  # CHANGED
print(f"After 1 step: {model.steps}")

model.my_step()  # CHANGED
print(f"After 2 steps: {model.steps}")