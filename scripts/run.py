import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.geometry.gear_profile import InvoluteGear
from matplotlib import pyplot as plt

if __name__ == "__main__":
    gear = InvoluteGear(module=2, teeth_num=15, thickness=10)
    profile = gear.generate_single_tooth_profile()

    plt.figure(figsize=(6, 6))
    plt.plot(
        profile[:, 0], profile[:, 1], 'b-', linewidth=2, label='Tooth Profile'
    )
    plt.plot([0, 0], [0, gear.r_a], 'k--', alpha=0.3, label='Symmetry Axis (Y)')

    # 绘制参考圆
    for r, label, color in [
        (gear.r_b, 'Base Circle', 'r'),
        (gear.r_p, 'Pitch Circle', 'g'),
        (gear.r_a, 'Addendum Circle', 'b'),
        (gear.r_f, 'Root Circle', 'm'),
    ]:
        circle = plt.Circle(
            (0, 0),
            r,
            fill=False,
            linestyle=':',
            alpha=0.5,
            label=label,
            color=color,
        )
        plt.gca().add_patch(circle)

    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.xlabel('X [mm]')
    plt.ylabel('Y [mm]')
    plt.legend(loc='upper right')
    plt.title('Single Involute Gear Tooth Profile')
    plt.tight_layout()
    plt.show()
