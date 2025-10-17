from model import CyberSlugModel
from agents import PreyAgent
import math

model = CyberSlugModel(num_slugs=1, hermi_population=20)
slug = model.cyberslugs[0]

# Place slug at center
model.space.move_agent(slug, (300, 300))
slug.angle = 0  # Facing right (0 degrees)

# Find a hermi and place it to the RIGHT of slug
hermi = None
for agent in model.schedule.agents:
    if isinstance(agent, PreyAgent) and agent.prey_type == 'hermi':
        hermi = agent
        break

if hermi:
    # Place hermi to the RIGHT of slug
    model.space.move_agent(hermi, (320, 300))  # Only 20 pixels away!

    # Let odors diffuse
    print("Diffusing odors...")
    for _ in range(30):
        model.update_odor_patches()

    print("=== TURNING TEST ===")
    print(f"Slug at: {slug.pos}, facing: {slug.angle}°")
    print(f"Hermi at: {hermi.pos}")
    print(f"Hermi is to the RIGHT\n")

    # Run steps
    for i in range(20):
        model.step()

        print(f"Step {i + 1}:")
        print(f"  Slug pos: ({slug.pos[0]:.1f}, {slug.pos[1]:.1f})")
        print(f"  Slug angle: {slug.angle:.1f}°")
        print(f"  Hermi sensors: L={slug.sns_odors_left[1]:.3f} R={slug.sns_odors_right[1]:.3f}")
        print(f"  Somatic map: {slug.somatic_map:.3f}")
        print(f"  Turn angle: {slug.turn_angle:.3f}")

        # Check direction
        if slug.turn_angle > 0:
            print(f"  ← Turning LEFT")
        elif slug.turn_angle < 0:
            print(f"  → Turning RIGHT")
        else:
            print(f"  | Going straight")

        if slug.hermi_counter > 0:
            print(f"\n✅✅✅ ATE THE HERMI! ✅✅✅")
            break
    else:
        print("\n❌ Did not eat hermi")
        print(f"Final distance: {math.sqrt((slug.pos[0] - hermi.pos[0]) ** 2 + (slug.pos[1] - hermi.pos[1]) ** 2):.1f}")