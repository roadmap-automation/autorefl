# autorefl — Autonomous Neutron Reflectometry

Framework for driving neutron reflectometry experiments autonomously. The goal is to measure a 1-D reflectometry pattern, *R(Q)*, with maximum efficiency by using Bayesian inference to decide where and for how long to measure next, rather than following a fixed scan plan.

## Background

Neutron reflectometry measures the reflection of neutrons from structured interfaces as a 1-D curve *R(Q)*. For a general reflectometer the measurement coordinate is a physical setting *x* (angle, energy, …) that maps to one or more *Q* values. After each measurement the posterior distribution over model parameters is updated, and the figure of merit (expected information gain per second) is evaluated at every candidate point to choose the next measurement.

The figure of merit is computed via differential entropy of the MCMC parameter draws; fitting uses DREAM via the [bumps](https://github.com/bumps/bumps) framework; reflectometry models are defined with [refl1d](https://github.com/reflectometry/refl1d).

---

## Installation

```bash
pip install -e /path/to/autorefl
```

[reductus](https://github.com/reductus/reductus) must be installed separately (it is excluded from `pyproject.toml` to avoid breaking editable installs when it is already present in a conda environment):

```bash
pip install git+https://github.com/reductus/reductus.git
```

### Other dependencies

| Package | Purpose |
|---------|---------|
| `bumps` | DREAM MCMC fitter |
| `refl1d` | Reflectometry model definition |
| `numpy`, `scipy`, `matplotlib` | Numerics and plotting |
| `dill` | Pickle extension for saving experiment state |
| `aiohttp` | Fit server and async HTTP client |
| `h5py` | HDF5 serialisation between fit server and client |
| `molgroups` | Molecular group models for `refl1d` (required by the example models) |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  AutoReflLauncher  (launch.py)                      │
│  asyncio event loop: fit → select → measure → loop  │
│                                                     │
│  ┌─────────────────┐   HTTP   ┌──────────────────┐  │
│  │  FitClient       │ ──────► │  fit_server      │  │
│  │  (fit_client.py) │ ◄────── │  (fit_server.py) │  │
│  └─────────────────┘  HDF5   └──────────────────┘  │
│            │                   (runs on HPC node    │
│            │                    or localhost)        │
│  ┌─────────┴──────────────┐                         │
│  │  AutoReflExperiment    │                         │
│  │  / SimReflExperiment   │                         │
│  │  (autorefl.py / simexp)│                         │
│  └────────────────────────┘                         │
│            │                                        │
│  ┌─────────┴──────────┐   ┌────────────────────┐    │
│  │  Instrument         │   │  NICECampaignTask  │    │
│  │  (instrument.py)    │   │  (remote/nicedata) │    │
│  └─────────────────────┘   └────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

**Fit server** — a standalone `aiohttp` process that runs DREAM and Q-profile calculations. It may run on an HPC node or alongside the main process. The main loop communicates with it over HTTP, sending the bumps problem (serialised with `dill`) and receiving the posterior draws and Q-profiles as an HDF5 byte stream.

**Experiment classes** — hold the MCMC state, step history, and figure-of-merit logic. They are instrument-agnostic; the `Instrument` object provides the calibration model.

**Launcher** — drives the async event loop: call fit server → select next point(s) via FOM → request measurement → receive data → repeat.

**Monitor** — optional `aiohttp`/SocketIO web server (`remote/monitor.py`) that streams live convergence and FOM plots to a browser.

---

## Class Hierarchy

```
AutoReflBase  (autorefl.py)
├── SimReflExperiment         (simexp.py)      — pure simulation, no hardware
│   └── SimReflExperimentControl               — fixed-scan control experiment
├── AutoReflExperiment        (autorefl.py)    — hardware measurement
└── PolarizedAutoReflExperiment (polarized.py) — polarized beam, hardware
    └── (PolarizedSimReflExperiment)           — polarized beam, simulation
```

| Class | Use |
|-------|-----|
| `SimReflExperiment` | Simulate an autonomous experiment offline. Takes ground-truth `bestpars` and generates synthetic count data. |
| `AutoReflExperiment` | Live experiment. `request_data()` produces `MeasurementPoint` objects to be dispatched to the NICE instrument queue. |
| `PolarizedAutoReflExperiment` | Like `AutoReflExperiment` but maps cross-section slots (mm, mp, pm, pp) onto the multi-model FOM loop. |
| `AutoReflLauncher` | Owns the `asyncio` event loop; drives fit → select → measure → ingest for hardware experiments. |

---

## Key Modules

| File | Contents |
|------|---------|
| `autorefl/autorefl.py` | `AutoReflBase`, `AutoReflExperiment` — core FOM + fit logic |
| `autorefl/simexp.py` | `SimReflExperiment`, `SimReflExperimentControl` |
| `autorefl/polarized.py` | Polarized cross-section subclasses |
| `autorefl/launch.py` | `AutoReflLauncher` — async hardware experiment loop |
| `autorefl/fit_server.py` | Standalone DREAM fit server (run as `python -m autorefl.fit_server`) |
| `autorefl/fit_client.py` | `FitClient` — HTTP client for the fit server |
| `autorefl/instrument.py` | `ReflectometerBase`, `MAGIK`, `CANDOR`, `LIQREF` — instrument digital twins |
| `autorefl/entropy.py` | Differential entropy estimators (`mvn_fast`, `mvn`, `gmm`) |
| `autorefl/datastruct.py` | `DataPoint`, `ExperimentStep`, `MeasurementPoint`, `Intent` |
| `autorefl/reduction.py` | Count-array → `ReflData` reduction (used internally; see note below) |
| `autorefl/simulation.py` | Poisson count simulation from a reflectometry model |
| `autorefl/calibration.py` | Intensity calibration from nexus files |
| `autorefl/analysis.py` | Plotting and analysis utilities for simulation studies |
| `autorefl/remote/nicedata.py` | `NICECampaignTask`, `Signaller` — NICE instrument API wrapper |
| `autorefl/remote/monitor.py` | Live SocketIO web monitor |
| `autorefl/remote/nicepath.py` | NICE path and file utilities |

---

## Running the Fit Server

The fit server must be started before any hardware or simulation run that uses `FitClient`:

```bash
python -m autorefl.fit_server --host 0.0.0.0 --port 5031
```

Environment variables:
- `AUTOREFL_FIT_HOST` — fit server hostname (default `127.0.0.1`)
- `AUTOREFL_FIT_PORT` — fit server port (default `5031`)

For simulation studies on a single machine, start the server in one terminal and run the experiment in another. For hardware experiments at NCNR, the server can run on a dedicated HPC node.

---

## Simulation Workflow

Simulation uses `SimulatedBackend`, which generates synthetic count data from the model's current parameter values as ground truth. The fit server must be running — simulation still uses DREAM for fitting, it just replaces hardware measurement with Poisson-sampled synthetic counts.

Start the fit server in one terminal, then run your simulation script:

```bash
python -m autorefl.fit_server
```

The script below runs end-to-end using the example model in `example_model/`. Copy it to the repo root (not inside `example_model/`) and run it:

```python
import asyncio
import os
import numpy as np
from pathlib import Path
from bumps.cli import load_model
from autorefl.autorefl import AutoReflExperiment
from autorefl.launch import AutoReflLauncher
from autorefl.backend import SimulatedBackend
from autorefl.fit_client import FitClient
from autorefl.remote.nicedata import Signaller
from autorefl import instrument

MODEL_DIR = Path(__file__).parent / 'example_model'

problem = load_model(str(MODEL_DIR / 'tiox_dopc_d2o.py'))

# Load ground-truth parameters — SimulatedBackend draws synthetic counts from these.
par_lines = (MODEL_DIR / 'tiox_dopc_d2o.par').read_text().splitlines()
bestpars = np.array([float(line.split()[-1]) for line in par_lines if line.strip()])

# Q grid
qmin, qmax, qstep, qstep_max = 0.008, 0.25, 0.0005, 0.0024
dq = np.linspace(qstep, qstep_max,
                 int(np.ceil(2 * (qmax - qmin) / (qstep_max + qstep))))
measQ = (qmin - qstep) + np.cumsum(dq)

instr = instrument.MAGIK()
exp = AutoReflExperiment(
    problem=problem,
    Q=measQ,
    instrument=instr,
    eta=0.68,
    meas_bkg=3e-6,
    bestpars=bestpars,
)
# For MAGIK the scan coordinate is Q directly (single wavelength, 1:1 with angle)
exp.x = exp.measQ

pathname = './tiox_sim_run'
os.makedirs(pathname, exist_ok=True)

signals = Signaller()
signals.global_start.set()  # skip the web-monitor Start-button gate

async def main():
    client = FitClient()
    await client.connect()
    try:
        launcher = AutoReflLauncher(
            exp, client, signals,
            maxtime=3600,
            pathname=pathname,
            backend=SimulatedBackend(exp),
        )
        await launcher.run()
    finally:
        await client.disconnect()

asyncio.run(main())
```

---

## Hardware Experiment Workflow

Hardware experiments run through `AutoReflLauncher` with a `NICEBackend`:

```python
from autorefl.autorefl import AutoReflExperiment
from autorefl.launch import AutoReflLauncher
from autorefl.fit_client import FitClient
from autorefl.remote.nicedata import NICECampaignTask, NICEBackend, Signaller
from autorefl import instrument

instr = instrument.CANDOR()
# ... calibrate_intensity(instr, ...) to load calibration files

exp = AutoReflExperiment(
    problem=problem,
    Q=Q,
    instrument=instr,
    eta=0.68,
    meas_bkg=1e-6,
)

pathname = '/data/my_experiment'
client = FitClient(host='hpc-node', port=5031)
signals = Signaller()
task = NICECampaignTask(instr.trajectoryMotors(), pathname)

launcher = AutoReflLauncher(
    exp=exp,
    client=client,
    signals=signals,
    maxtime=43200,
    pathname='/data/my_experiment',
    backend=NICEBackend(task),
)

await asyncio.gather(
    asyncio.to_thread(api.serve_tasks, task),
    launcher.run(),
)
```

For polarized beam experiments, substitute `PolarizedAutoReflExperiment` and pass `polarization_states` (mapping from xs name to flipper motor positions).

---

## Extending to Custom Hardware

Two things are needed: a custom `Instrument` subclass and a `MeasurementBackend` subclass. The FOM loop in `AutoReflBase` is entirely independent of hardware.

### 1. Instrument subclass

The `Instrument` object is not just metadata — `intensity(x)` feeds directly into the FOM calculation (expected counts per second scales the information-gain estimate). A wrong intensity model will cause the algorithm to make poor time-allocation decisions.

**Multi-detector measurement time:** For instruments where all detector channels measure *simultaneously* at one angle setting (e.g. CANDOR with 54 wavelength channels), the physical hold time at an angle is determined by the slowest (hardest) channel — i.e. `max(t_k)` over channels, not a per-channel average. The FOM still sums `dH/dt` contributions across channels, but the measurement time returned to the scheduler and used by the simulator is the channel-wise maximum. This is handled automatically; custom `Instrument` subclasses do not need to implement anything extra, but calibration data returned by `intensity(x)` must accurately represent per-channel rates so that high-Q channels (low rate → long time) are not under-timed.

Required methods:

| Method | Returns | Used for |
|--------|---------|---------|
| `intensity(x)` | `ndarray` shape `(len(x), n_detectors)` | FOM: expected neutron rate |
| `x2q(x)` | `ndarray` same shape | FOM: Q values at each position |
| `qrange2xrange(qmin, qmax)` | `(xmin, xmax)` tuple | Setting up `exp.x` grid |
| `movetime(x)` | `float` seconds | FOM: movement penalty |
| `trajectoryData(x, intent)` | `dict` | Populating `MeasurementPoint.movements` |
| `Q2TdTLdL(Q, x, Q_rebin)` | `(T, dT, L, dL)` tuples | Fit server Q-profile calculation |

`trajectoryData` and `movetime` can return `{}` and `0.0` respectively as stubs if movement penalties are not needed. `Q2TdTLdL` is required by the fit server; for a monochromatic instrument it reduces to angle = arcsin(Qλ/4π) at a single wavelength.

### 2. MeasurementBackend subclass

Subclass `MeasurementBackend` from `autorefl.backend` and implement `measure()`. Pass it to `AutoReflLauncher` via the `backend` constructor parameter.

```python
from autorefl.backend import MeasurementBackend
from autorefl.datastruct import DataPoint

class MyBackend(MeasurementBackend):

    def __init__(self, my_hardware) -> None:
        self.hw = my_hardware
        self._last_x: float | None = None

    @property
    def last_x(self) -> float | None:
        return self._last_x

    async def measure(
        self,
        points: list[list[MeasurementPoint]],
    ) -> dict[int, list[DataPoint]]:
        data: dict[int, list[DataPoint]] = {}
        for ptlist in points:
            for mp in ptlist:
                # mp.base.x      — target position (degrees for MAGIK, angle for CANDOR)
                # mp.base.t      — count time in seconds
                # mp.base.intent — Intent.spec / Intent.backp / Intent.backm
                # mp.base.model  — model index (0 for single-model problems)
                T, dT, L, dL, N, Ninc = await self.hw.count(mp.base.x, mp.base.t)
                dp = DataPoint(mp.base.x, mp.base.t, mp.base.model,
                               (T, dT, L, dL, N, Ninc),
                               intent=mp.base.intent, movet=0.0)
                self._last_x = mp.base.x
                data.setdefault(mp.step_id, []).append(dp)
        return data

    def stop(self) -> None:
        self.hw.stop()


signals = Signaller()
signals.global_start.set()   # skip the web-monitor start-button gate if not using it

launcher = AutoReflLauncher(
    exp, client, signals, maxtime=43200, pathname='./run',
    backend=MyBackend(my_hardware),
)
asyncio.run(launcher.run())
```

**Physics fields:** `T`, `dT`, `L`, `dL`, `N`, `Ninc` must contain real measured values in physical units (degrees, Ångströms, counts) — they are passed directly to reductus for reduction.

**`last_x`:** returning the most recently measured position from this property keeps the movement-penalty calculation accurate. Return `None` if movement penalties are not used.

**`stop()`:** called by `AutoReflLauncher.stop()` on cancellation. Override to stop hardware cleanly.

**Web monitor:** `AutoReflLauncher.run()` starts the live R(Q)/FOM monitor automatically on port 5006. `signals.global_start` is set by its Start button; call `signals.global_start.set()` directly to skip it.

---

## Instruments

| Class | Instrument | Notes |
|-------|-----------|-------|
| `MAGIK` | MAGIK monochromatic reflectometer | Single bank, point detector |
| `CANDOR` | CANDOR polychromatic reflectometer | Multi-bank, multi-detector |
| `LIQREF` | LIQREF time-of-flight reflectometer | TOF, ragged Q per angle |

Instrument objects provide: `intensity(x)`, `x2q(x)`, `movetime(x)`, `trajectoryData(x)`, and `Q2TdTLdL(Q, x, Q_rebin)`. Intensity calibration is loaded from nexus files via `calibrate_intensity()`.

---

## Entropy Methods

The figure of merit is the expected information gain *dH/dt* (entropy change per second). Three entropy estimators are available via `entropy_options`:

| Method | Description |
|--------|-------------|
| `mvn_fast` | Multivariate normal approximation (default; fast) |
| `mvn` | Exact MVN via log-determinant |
| `gmm` | Gaussian mixture model via sklearn (slowest; most accurate for multimodal posteriors) |

---

## Known Limitations and Open Work

- **Reduction abstraction gap:** `reduction.py` uses a hand-rolled count-assembly path (`DataPoint → DataPoint2ReflData`). The authoritative reduction used elsewhere in the ROADMAP pipeline uses `reductus.reduce_from_files`. Until these are unified, hardware-included simulation mode and coherent archival are not possible. See `TODO: AutoRefl Reduction Abstraction` in `PROGRESS.md`.

- **Single-model hardware path:** `AutoReflExperiment` (unpolarized) is tested with single-model problems. Multi-model switching in hardware mode is implemented but not yet hardware-validated.

- **Model selection:** Autonomous switching between bumps models with different parameter spaces (e.g. choosing between 1-layer and 2-layer fits) is designed but not yet implemented. See `.claude/plans/autorefl-model-selection.md`.

- **MAGIK support:** `DummySampleChanger` and a `magik_app.py` entry point for the MAGIK instrument are planned but not yet written. The reflectometry method classes are already instrument-agnostic.

- **FOM ranking vs. actual time cost (open):** The FOM sums `dH/dt` per channel using each channel's individual `t_k` as the denominator, but the actual time cost of an angle setting is `max(t_k)`. This means the FOM slightly overvalues angle settings where some channels are easy and some are hard — the easy channels inflate `sum(dH/dt)` but the time cost is set by the hard channels. The simulation time passed to the scheduler correctly uses `max(t_k)` (fixed 2026-10-24); the ranking bias is a secondary issue that does not affect measurement validity.

---

## Example Models

`example_model/` contains two TiO₂-supported DOPC bilayer models:

| File | Description |
|------|-------------|
| `tiox_dopc_d2o.py` | Single-contrast model in D₂O |
| `tiox_dopc_d2o.par` | Ground-truth parameters for the D₂O model |
| `tiox_dopc.py` | Two-contrast model (H₂O + D₂O simultaneously) |
| `tiox_dopc.par` | Ground-truth parameters for the two-contrast model |
| `ch060_h2o_ph7.refl` | Measured H₂O contrast data (used by `tiox_dopc.py`) |
| `ch061_d2o_ph7.refl` | Measured D₂O contrast data (used by `tiox_dopc.py`) |

These models use `molgroups` — ensure it is installed in the same environment as the fit server.

Pass `--pars` to set the ground truth for simulation. Omit it to use the model's current parameter values.

Use `--sel` to restrict the figure of merit to a subset of parameters by index (e.g. `--sel 3 4 5`). By default all free parameters are included. Run `bumps --edit tiox_dopc_d2o.py` to inspect parameter indices.
