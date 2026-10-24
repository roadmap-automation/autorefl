"""Autonomous reflectometry simulation — end-to-end example.

Runs a complete AutoRefl campaign in simulation mode: no hardware is required.
Synthetic count data is generated from the model's ground-truth parameters via
Poisson sampling, exactly as the real instrument would produce, so the FOM loop
and convergence behaviour are representative of a live experiment.

Prerequisites
-------------
1. The fit server must be running in a separate terminal before launching this
   script.  It handles DREAM MCMC fitting and Q-profile calculations
   (default port 5031):

       python -m autorefl.fit_server

   Override host/port with env vars if needed:
       AUTOREFL_FIT_HOST=127.0.0.1  AUTOREFL_FIT_PORT=5031

2. The example model uses molgroups — install it in the same environment:

       pip install molgroups

Usage
-----
Set the INSTR constant below to 'CANDOR' or 'MAGIK', then run:

    python example_refl_sim.py

Output is written to ./{instr}_sim_run/ (created if absent).  The live web
monitor starts automatically on port 5006 — open it in a browser to watch
convergence and the FOM map in real time.  The simulation runs until maxtime
seconds of simulated beam time have elapsed.

Tuning
------
Key parameters to adjust for your own model:

  INSTR          'CANDOR' (54-channel polychromatic) or 'MAGIK' (monochromatic)
  npoints        candidate points evaluated per FOM step (more = slower but
                 broader exploration; 2-6 is typical)
  eta            posterior CI width target (0-1); higher = more conservative,
                 measures longer per point
  meas_bkg       incoherent background floor (counts/s); affects low-signal
                 time estimates at high Q
  oversampling   probe oversampling factor for resolution smearing in the fit;
                 11 is accurate, lower values are faster
  fit_options    DREAM settings: burn (equilibration steps), steps (production
                 steps), pop (population multiplier x nparams), alpha (tempering)
  maxtime        wall-clock limit in seconds for the campaign
"""

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

# ── Instrument selection ───────────────────────────────────────────────────────
# 'CANDOR' — 54-channel polychromatic reflectometer; x = angle (degrees)
# 'MAGIK'  — single-wavelength monochromatic reflectometer; x = Q (Angstroms^-1)
INSTR = 'CANDOR'

# ── Model ─────────────────────────────────────────────────────────────────────
# tiox_dopc_d2o: TiO2-supported DOPC bilayer in D2O, single contrast.
# The .par file holds the ground-truth parameter values used by SimulatedBackend
# to generate synthetic counts.  Swap these for your own model and par file.
MODEL_DIR = Path(__file__).parent / 'example_model'

problem = load_model(str(MODEL_DIR / 'tiox_dopc_d2o.py'))

par_lines = (MODEL_DIR / 'tiox_dopc_d2o.par').read_text().splitlines()
bestpars = np.array([float(line.split()[-1]) for line in par_lines if line.strip()])

# ── Q grid (measQ) ────────────────────────────────────────────────────────────
# Non-uniform spacing: finer at low Q (sharp features), coarser at high Q
# (featureless decay).  The cumsum of a linspace gives a quadratic-ish grid
# that allocates more points where dR/dQ is largest.
# measQ defines the rebinning axis used by the fit server and FOM calculation —
# it is NOT the list of angles to visit; the instrument's x grid is derived
# from it separately below.
qmin, qmax, qstep, qstep_max = 0.008, 0.25, 0.0005, 0.0024
dq = np.linspace(qstep, qstep_max,
                 int(np.ceil(2 * (qmax - qmin) / (qstep_max + qstep))))
measQ = (qmin - qstep) + np.cumsum(dq)

# ── Instrument and experiment ──────────────────────────────────────────────────
instr = getattr(instrument, INSTR)()

exp = AutoReflExperiment(
    problem=problem,
    Q=measQ,
    instrument=instr,
    npoints=3,           # candidate points per FOM step
    eta=0.5,             # posterior CI width target
    meas_bkg=3e-6,       # incoherent background floor (counts/s)
    bestpars=bestpars,   # ground truth for SimulatedBackend; omit for hardware
    oversampling=11,     # resolution-smearing oversampling in fit
    fit_oversampling=False,
    fit_options={'burn': 1000, 'steps': 100, 'pop': 4, 'alpha': 0.000},
)

# ── Scan coordinate (exp.x) ───────────────────────────────────────────────────
# exp.x is the grid of instrument positions the FOM is evaluated over.
# For CANDOR (polychromatic), __init__ builds a sensible angle linspace via
# qrange2xrange — no override needed.
# For MAGIK (monochromatic, x = Q), setting x = measQ exactly aligns the scan
# grid with the rebinning axis.
if INSTR == 'MAGIK':
    exp.x = exp.measQ

# ── Output directory ──────────────────────────────────────────────────────────
pathname = f'./{INSTR.lower()}_sim_run'
os.makedirs(pathname, exist_ok=True)

# ── Signals ───────────────────────────────────────────────────────────────────
# In hardware mode, global_start is set by the Start button in the web monitor.
# In simulation mode we set it immediately to skip the gate.
signals = Signaller()
signals.global_start.set()

# ── Main loop ─────────────────────────────────────────────────────────────────
async def main():
    # FitClient connects to the fit server (default: localhost:5031).
    # Override with AUTOREFL_FIT_HOST / AUTOREFL_FIT_PORT env vars if needed.
    client = FitClient()
    await client.connect()
    try:
        launcher = AutoReflLauncher(
            exp, client, signals,
            maxtime=3600,                   # simulated beam-time budget (seconds)
            pathname=pathname,
            backend=SimulatedBackend(exp),  # replace with NICEBackend for hardware
        )
        await launcher.run()
    finally:
        await client.disconnect()

asyncio.run(main())
