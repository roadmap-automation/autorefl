"""Polarized beam AutoRefl experiment class.

PolarizedAutoReflExperiment extends AutoReflExperiment by mapping
cross-sections (xs) onto the existing nmodels dimension.  Each active xs slot
becomes one AutoRefl "model"; switch_penalty=0.0 makes xs transitions
cost-free so the FOM loop allocates time across xs freely.

Key differences from AutoReflExperiment:
  - __init__: expand nmodels=1 → n_active_xs, build xs_indices mapping
  - update_models: dispatch to probe.xs[xs_indices[i]] instead of models[i].probe
  - request_data: merge flipper motor positions into each MeasurementPoint
  - _simulate_datapoint: index into calc_expected_R result by model_num

Use with SimulatedBackend for offline simulation (no hardware required).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from bumps.fitproblem import FitProblem
from reductus.reflred.intent import Intent

from . import instrument as instrument_module
from .autorefl import AutoReflExperiment, default_fit_options, default_entropy_options
from .datastruct import DataPoint, MeasurementPoint
from .reduction import reduce
from .simulation import sim_data_N, calc_expected_R, _FWHM

# Cross-section name order matches PolarizedNeutronProbe._xs_names
_XS_NAMES: List[str] = ['mm', 'mp', 'pm', 'pp']


class PolarizedAutoReflExperiment(AutoReflExperiment):
    """AutoRefl experiment for polarized neutron reflectometry.

    Treats each active cross-section as an AutoRefl model.  The bumps
    FitProblem must contain exactly one Experiment whose probe is a
    PolarizedNeutronProbe.  Cross-sections that are None in the probe
    (half-polarized case) are silently excluded from the model list.

    Args:
        filename: NICE file prefix for data collection.
        problem: Bumps FitProblem with a single polarized Experiment.
        Q: Q grid (single array broadcast to all xs).
        instrument: Reflectometer digital twin; polarization_states must
            be set on it before passing here.
        polarization_states: Mapping from xs name to flipper motor positions,
            e.g. ``{'pp': {'frontFlipper': 1, 'backFlipper': 1}, ...}``.
            Must contain an entry for every xs name in _XS_NAMES, even if
            that xs is None in the probe (unused entries are never accessed).
        **kwargs: Forwarded to AutoReflExperiment (eta, npoints, meas_bkg, …).
    """

    def __init__(
        self,
        problem: FitProblem,
        Q: np.ndarray,
        instrument: instrument_module.ReflectometerBase,
        polarization_states: Dict[str, dict],
        **kwargs,
    ) -> None:
        models = list(problem.models)
        assert len(models) == 1, (
            "PolarizedAutoReflExperiment requires a FitProblem with exactly one model"
        )
        assert hasattr(models[0].probe, 'xs'), (
            "PolarizedAutoReflExperiment requires a PolarizedNeutronProbe"
        )

        # switch_penalty=1.0: all xs treated equally (no penalty for switching).
        # Values >1 would bias toward staying on the current xs; 0.0 causes div/0.
        kwargs.setdefault('switch_penalty', 1.0)
        super().__init__(problem, Q, instrument, **kwargs)

        self.polarization_states = polarization_states

        # Build xs_indices: AutoRefl model index i → probe.xs index
        probe = models[0].probe
        self.xs_indices: List[int] = [
            i for i, xs in enumerate(probe.xs) if xs is not None
        ]
        n = len(self.xs_indices)
        assert n > 0, "PolarizedNeutronProbe has no active cross-sections"

        # Expand all per-model arrays from 1 → n
        base_model = self.models[0]
        self.models = [base_model] * n
        self.nmodels = n

        base_Q = np.asarray(self.measQ[0])
        self.measQ = np.broadcast_to(base_Q, (n, len(base_Q))).copy()
        self.x = [self.x[0].copy() for _ in range(n)]
        self.meas_bkg = np.full(n, self.meas_bkg[0])
        self.resid_bkg = np.full(n, self.resid_bkg[0])

    def update_models(self) -> None:
        modeldata = self.get_data()
        experiment = list(self.problem.models)[0]

        for i, (measQ, (specdata, bkgpdata, bkgmdata)) in enumerate(
            zip(self.measQ, modeldata)
        ):
            xs_probe = experiment.probe.xs[self.xs_indices[i]]
            if xs_probe is None:
                continue
            spec = reduce(specdata, bkgpdata, bkgmdata)
            if spec is None:
                continue
            mT = spec.sample.angle_x
            mdT = spec.angular_resolution
            mL = spec.detector.wavelength
            mdL = spec.detector.wavelength_resolution
            mR = spec.v
            mdR = spec.dv
            mQ = spec.Qz
            mdQ = spec.dQ
            xs_probe._set_TLR(mT, mdT * _FWHM, mL, mdL * _FWHM, mR, mdR, dQ=mdQ)
            xs_probe._apply_oversamplings()
            xs_probe.resolution = self.instrument.resolution

        experiment.update()
        self.problem.partial = True
        self.problem.model_reset()
        self.problem.chisq_str()

    def request_data(self, newpoints, foms) -> List[List[MeasurementPoint]]:
        point_lists = super().request_data(newpoints, foms)
        for point_list in point_lists:
            for mp in point_list:
                xs_name = _XS_NAMES[self.xs_indices[mp.base.model]]
                for motor, val in self.polarization_states[xs_name].items():
                    mp.movements += [motor, val]
        return point_lists

    def _simulate_datapoint(
        self,
        model_num: int,
        x: float,
        t: float,
        intent: 'Intent',
        merit: Optional[float],
    ) -> 'DataPoint':
        """Simulate a single measurement point for the polarized case.

        Re-sets the probe T/L at the measurement angle x (so the Q grid matches
        the instrument channels, not the measQ rebinning axis), then picks the
        R array for the cross-section corresponding to model_num.
        """
        T = self.instrument.T(x)[0]
        dT = self.instrument.dT(x)[0]
        L = self.instrument.L(x)[0]
        dL = self.instrument.dL(x)[0]
        intens = self.instrument.intensity(x)[0]

        incident_neutrons = intens * t

        # calc_expected_R returns one R per active xs in xs_indices order.
        calcR = calc_expected_R(
            self.calcmodels[0], T, dT, L, dL,
            self.oversampling, self.instrument.resolution
        )[model_num]

        Nspec, (Nbkg, _), Ninc = sim_data_N(
            calcR, incident_neutrons,
            self.resid_bkg[model_num],
            self.meas_bkg[model_num],
        )

        if intent == Intent.slit:
            N = Ninc
        elif intent in (Intent.backp, Intent.backm):
            N = Nbkg
        else:
            N = Nspec

        return DataPoint(x, t, model_num, (T, dT, L, dL, N, Ninc),
                         intent=intent, merit=merit)
