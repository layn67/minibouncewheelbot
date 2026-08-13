import argparse
from pathlib import Path

from wheelbot.experiments import EXPERIMENTS
from wheelbot.simulation import Simulation


def summarize(rows):
    final = rows[-1]
    peak_tilt = max(max(abs(row["roll"]), abs(row["pitch"])) for row in rows)
    return (
        f"final x={final['x']:.3f} m, "
        f"roll={final['roll']:.3f} rad, "
        f"pitch={final['pitch']:.3f} rad, "
        f"yaw={final['yaw']:.3f} rad, "
        f"peak tilt={peak_tilt:.3f} rad, "
        f"phase={final['phase']}"
    )


def run(name, render):
    experiment = EXPERIMENTS[name]()
    output = Path("results") / f"{name}.csv"
    rows = Simulation().run(experiment, render=render, output=output)
    print(f"{name}: {summarize(rows)}")
    print(f"saved {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the lean Wheelbot control experiments."
    )
    parser.add_argument("experiment", nargs="?", choices=[*EXPERIMENTS, "all"])
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    if args.experiment is None:
        for name in EXPERIMENTS:
            print(name)
        return
    if args.experiment == "all":
        for name in EXPERIMENTS:
            run(name, False)
        return
    run(args.experiment, args.render)


if __name__ == "__main__":
    main()
