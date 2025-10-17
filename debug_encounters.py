# Create file: debug_encounters.py

from model import CyberSlugModel
import math

model = CyberSlugModel(
    num_slugs=1,
    hermi_population=20,
    width=600,
    height=600
)

slug = model.cyberslugs[0]

print("=== INITIAL STATE ===")
print(f"Slug position: {slug.pos}")
print(f"Slug size: {slug.size}")
print(f"Number of prey in model: {sum(1 for a in model.schedule.agents if hasattr(a, 'prey_type'))}")

# Run for 100 steps and track distances
for i in range(100):
    model.step()

    # Find closest hermi
    from agents import PreyAgent

    min_dist = float('inf')
    closest_prey = None

    for agent in model.schedule.agents:
        if isinstance(agent, PreyAgent) and agent.prey_type == 'hermi':
            px, py = agent.pos
            sx, sy = slug.pos
            dist = math.sqrt((px - sx) ** 2 + (py - sy) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest_prey = agent

    if i % 20 == 0:
        print(f"\nStep {i}:")
        print(f"  Slug pos: ({slug.pos[0]:.1f}, {slug.pos[1]:.1f})")
        print(f"  Closest hermi distance: {min_dist:.1f}")
        print(f"  Hermi eaten: {slug.hermi_counter}")
        print(f"  Encounter range: {0.4 * slug.size:.1f}")

        # Check if should have encountered
        if min_dist < (0.4 * slug.size):
            print(f"  ⚠️ SHOULD HAVE ENCOUNTERED! (dist={min_dist:.1f} < range={0.4 * slug.size:.1f})")

print(f"\n=== FINAL ===")
print(f"Total Hermi eaten: {slug.hermi_counter}")
print(f"Slug moved: {slug.pos}")