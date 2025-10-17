# test_turning.py

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
    model.space.move_agent(hermi, (350, 300))  # 50 pixels to the right

    print("=== TURNING TEST ===")
    print(f"Slug at: {slug.pos}, facing: {slug.angle}°")
    print(f"Hermi at: {hermi.pos}")
    print(f"Hermi is to the RIGHT")

    # Run a few steps and watch turning
    for i in range(10):
        model.step()

        print(f"\nStep {i + 1}:")
        print(f"  Slug pos: ({slug.pos[0]:.1f}, {slug.pos[1]:.1f})")
        print(f"  Slug angle: {slug.angle:.1f}°")
        print(f"  Hermi_left sensor: {slug.sns_odors_left[1]:.3f}")
        print(f"  Hermi_right sensor: {slug.sns_odors_right[1]:.3f}")
        print(f"  Somatic map: {slug.somatic_map:.3f}")
        print(f"  Turn angle: {slug.turn_angle:.3f}")
        print(f"  AppState: {slug.app_state:.3f}")

        # Check if turning toward or away
        angle_to_hermi = math.degrees(math.atan2(hermi.pos[1] - slug.pos[1],
                                                 hermi.pos[0] - slug.pos[0]))
        angle_diff = abs((angle_to_hermi - slug.angle + 180) % 360 - 180)

        if angle_diff < 90:
            print(f"  ✅ Facing TOWARD hermi (diff: {angle_diff:.1f}°)")
        else:
            print(f"  ❌ Facing AWAY from hermi (diff: {angle_diff:.1f}°)")

        # Stop if we ate it
        if slug.hermi_counter > 0:
            print("\n✅ ATE THE HERMI!")
            break

    if slug.hermi_counter == 0:
        print("\n❌ NEVER ATE THE HERMI - AVOIDING IT!")