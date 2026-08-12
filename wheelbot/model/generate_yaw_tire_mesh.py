from __future__ import annotations

import math
from pathlib import Path


def main() -> None:
    major_radius = 0.027
    section_radius = 0.005
    around_segments = 96
    section_segments = 32
    output = Path(__file__).resolve().parent / "wheel_tire.obj"

    lines: list[str] = ["# Smooth rounded custom yaw tyre"]
    for around in range(around_segments):
        theta = 2.0 * math.pi * around / around_segments
        cosine = math.cos(theta)
        sine = math.sin(theta)
        for section in range(section_segments):
            phi = 2.0 * math.pi * section / section_segments
            radial = major_radius + section_radius * math.cos(phi)
            x = radial * cosine
            y = section_radius * math.sin(phi)
            z = radial * sine
            lines.append(f"v {x:.9f} {y:.9f} {z:.9f}")

    def vertex(around: int, section: int) -> int:
        return (around % around_segments) * section_segments + (section % section_segments) + 1

    for around in range(around_segments):
        for section in range(section_segments):
            a = vertex(around, section)
            b = vertex(around + 1, section)
            c = vertex(around + 1, section + 1)
            d = vertex(around, section + 1)
            lines.append(f"f {a} {b} {c}")
            lines.append(f"f {a} {c} {d}")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
