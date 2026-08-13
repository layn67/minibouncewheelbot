# Wheelbot: a self-balancing robot with spring-launched jump actuation

Simulation code for the MSc project *Design, Dynamic Modelling and Control of a
Self-Balancing Robot with Jump Actuation* (MECH5845M, School of Mechanical
Engineering, University of Leeds). Section, table and appendix numbers used
throughout this README refer to the project report, which is not distributed with
the code.

The plant is a 0.688 kg reaction-wheel unicycle: two mutually orthogonal wheels
either of which can be the ground wheel, split into two halves by a central
prismatic joint carrying a 6000 N/m compression spring. One wheel rolls while the
other acts as a reaction wheel, so both unstable axes are directly actuated. It
balances, translates, turns without a yaw actuator, clears rectangular obstacles
by jumping, and rights itself after a fall. Everything is simulated in MuJoCo; no
hardware was built.

## Quick start

```bash
python -m venv .env && . .env/bin/activate && pip install -e .
```

```bash
python -m wheelbot.cli all && python -m wheelbot.validate_poles && python -m wheelbot.jump_validation && python -m wheelbot.plots
```

`cli all` writes eleven experiment logs to `results/*.csv`, `validate_poles` adds
the two open-loop divergence logs, `jump_validation` adds the launch-map
identification, and `plots` turns all of it into the seventeen figures in
`figures/`. Run them in that order: a missing input makes `plots` skip a figure or
draw a reduced version of it rather than fail. Add `--render` to a single
experiment to watch it in the MuJoCo viewer:

```bash
python -m wheelbot.cli unified --render
```

Requires Python 3.11+, `mujoco`, `numpy` and `matplotlib`, all pulled in by
`pip install -e .`.

## What runs what

Every number in the report comes from one of these commands, and no module reads
a result another module did not write (Appendix G.3). The one exception is
`fig_sensitivity`, because `sensitivity` prints its 54-cell outcome grid to the
terminal instead of writing a file, so the figure carries that grid as a literal.
Re-run `wheelbot.sensitivity` to check the figure against it.

| Command | Produces |
| --- | --- |
| `python -m wheelbot.cli all` | The eleven experiment logs in `results/` |
| `python -m wheelbot.linear_model` | Reduced-model matrices, poles, controllability, actuator envelope (Appendix C) |
| `python -m wheelbot.validate_poles` | Predicted vs measured open-loop divergence (Section 3.6) |
| `python -m wheelbot.jump_validation` | Launch-map identification and the three-obstacle campaign (Appendix E.2, E.7) |
| `python -m wheelbot.jump_feasibility` | Feasibility envelopes, margins, in-flight attitude (Appendix E.5, E.8) |
| `python -m wheelbot.yaw_identification` | The steering-coefficient trials reported as a failure (Section 4.6) |
| `python -m wheelbot.sensitivity` | The 54-cell ±20 % parameter sweep (Section 6.5) |
| `python -m wheelbot.plots` | The seventeen numbered figures in `figures/` |

`plots` simulates nothing itself. It reads the CSVs written by `cli all`,
`val_roll.csv` and `val_pitch.csv` from `validate_poles`, and
`jump_launch_identification.csv` from `jump_validation`, so all three must run
before it.
`jump_feasibility` writes its own envelope figures straight into `results/`;
`figures/jump_feasibility_26mm.png` is a copy of one of them. Every other module
is self-contained.

The committed `figures/` are exactly what these commands produce, so they can be
regenerated and diffed.

The eleven experiments are `passive_fall`, `pitch_balance`, `roll_balance`,
`motion`, `disturbance`, `yaw`, `yaw_wheel_x`, `self_right_pitch`,
`self_right_roll`, `jump` and `unified`. Run `python -m wheelbot.cli` with no
argument to list them. `pip install -e .` also puts the same runner on the path
as `wheelbot`, so `wheelbot all` and `python -m wheelbot.cli all` are the same
command.

## Module map

| Module | Role |
| --- | --- |
| `physical_design.py` | The single constant set: masses, inertias, radii, spring rate, actuator limits |
| `model/wheelbot.xml` | The single plant definition (MJCF) |
| `linear_model.py` | Reduced pitch and roll models, poles, controllability, actuator envelope, barrier model |
| `self_right_model.py` | Barrier constants shared by the controller and the analysis |
| `jump_planner.py` | Minimum-energy planner, clearance geometry, convex solve |
| `control/` | The control laws: `balance`, `yaw`, `jump`, `self_right`, `unified`, `common` |
| `roles.py` | Role-coordinate mapping and contact debounce |
| `experiments.py`, `disturbances.py` | Experiment definitions and the disturbance campaign |
| `simulation.py` | Integration loop, actuator envelope, telemetry logging |
| `plots.py` | Every figure |

`physical_design.py` is the load-bearing one. The MJCF, the reduced models, the
controller gains and the planner all take their constants from it, so a parameter
cannot drift between a model and the thing it models. Section 3.7 explains why
that is a structural safeguard rather than tidiness.

## What the results are

| Capability | Result |
| --- | --- |
| Reduced models | Predicted divergence 14.686 and 8.961 rad/s on pitch and roll, measured 13.89 and 8.70, errors -5.4 % and -2.9 % |
| Balancing | Monotonic recovery from 50° pitch and 20° roll; settling to 0.02 rad in 4.80 s and 3.42 s |
| Translation | Five waypoints in 18.7 s, 36 mm residual error, peak tilt 0.102 rad |
| Heading | 180° turn in 48.5 s at a mean 0.065 rad/s; momentum-duty-cycle limited, not torque limited |
| Jumping | 26, 35 and 42 mm obstacles cleared with 12.16-15.13 mm sampled clearance and zero contacts |
| Self-righting | 2.68 s from 80° pitch, 2.42 s from 100° roll, using 82-85 % of the wheel-speed ceiling |
| Integration | Fallen start recovered, 90° turn, 26 mm obstacle cleared, stopped at 1.498 m against 1.500 m, in 44.22 s |
| Sensitivity | 48 of 54 cells pass under ±20 % perturbation of nine parameters |

Each capability is limited by a different quantity, and only one is the obvious
one. Balancing is limited by the attitude envelope, jumping by how mass divides
between the two chassis halves rather than by the spring, self-righting and
disturbance rejection by wheel speed. Heading is limited by none of these but by
a momentum budget, the one quantity the design cannot buy more of, and the one
that consumed 63 % of the integrated mission.

## What this does not establish

The study is simulation-only and no prototype was built. Every experiment is a
single deterministic trajectory with exact state feedback: no random seeding, no
sensor model, no measurement noise, no actuation latency, no repetition, so no
error bars are quoted anywhere. Parameters are chosen rather than measured: they
are internally consistent and lie in the range occupied by published machines of
this scale, but nothing here predicts a specific built robot. The actuator
envelope is optimistic: flat torque to a 7000 rpm corner implies one wheel drawing
over four fifths of the power the comparable real machine supplies to both motors,
so results needing sustained high rotor speed are upper bounds. The jump rests on
a simulated elastic impact, the regime Acosta et al. (2022) found simulators
reproduce least well and MuJoCo worst of the three they tested. No control-method
superiority is claimed, because none was tested. Sections 3.7, 5.8, 6.6 and 7.2
of the report state each of these in full.
