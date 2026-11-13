"""Measurement backend abstraction for AutoReflLauncher.

A MeasurementBackend is the single extension point for connecting
AutoReflLauncher to a hardware or simulation data source.  Subclass it
and implement measure_position().

Built-in backends
-----------------
SimulatedBackend  — generates synthetic count data from a SimReflExperiment;
                    used by sim_app.py and runauto_cli.py
NICEBackend       — wraps NICECampaignTask for NCNR NICE-controlled instruments;
                    defined in autorefl.remote.nicedata alongside the task class
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .datastruct import DataPoint, MeasurementPoint
    from .autorefl import AutoReflBase


class MeasurementBackend:
    """Base class for measurement hardware backends.

    Subclasses must implement :meth:`measure_position`.  Override :meth:`stop` to
    clean up hardware on cancellation.  The ``last_x`` property should
    return the most recently measured x position so the launcher can keep
    ``exp.instrument.x`` current for movement-penalty calculations; return
    ``None`` if the concept does not apply to your hardware.
    """

    @property
    def last_x(self) -> float | None:
        """Most recently measured x position, or None."""
        return None

    @property
    def campaign_file_info(self) -> 'dict | None':
        """FileInfo dict for the current campaign NeXus snapshot, or None.

        NICEBackend overrides this to return a snapshot path for reductus reduction.
        SimulatedBackend and other backends return None (DataPoint path used instead).
        """
        return None

    async def measure_position(
        self,
        pos: list['MeasurementPoint'],
    ) -> list['DataPoint']:
        """Measure one x-position (all intents) and return the acquired data.

        Parameters
        ----------
        pos:
            All :class:`~autorefl.datastruct.MeasurementPoint` objects for a
            single x value — typically one specular point plus up to two
            background points (background± at the same position).

        Returns
        -------
        list of DataPoint
            Returned DataPoints must have ``T``, ``dT``, ``L``, ``dL``,
            ``N``, ``Ninc`` populated with measured values in physical
            units (degrees, Ångströms, counts).
        """
        raise NotImplementedError

    def terminate_count(self) -> None:
        """Signal the backend to end the current count early and accept partial data.  No-op by default."""

    def skip_position(self) -> None:
        """Terminate the current count and skip remaining intents for this position.  No-op by default."""

    def stop(self) -> None:
        """Signal the backend to stop cleanly.  Called by AutoReflLauncher.stop()."""


class SimulatedBackend(MeasurementBackend):
    """Backend that generates synthetic count data from a reflectometry experiment.

    Uses ``exp._simulate_datapoint`` with the model's current parameter
    values as ground truth.  Works with any :class:`~autorefl.autorefl.AutoReflBase`
    subclass that implements ``_simulate_datapoint``.

    Parameters
    ----------
    exp:
        A :class:`~autorefl.autorefl.AutoReflBase` instance (typically
        :class:`~autorefl.simexp.SimReflExperiment`).
    """

    def __init__(self, exp: 'AutoReflBase') -> None:
        self.exp = exp
        self._last_x: float | None = None

    @property
    def last_x(self) -> float | None:
        return self._last_x

    async def measure_position(
        self,
        pos: list['MeasurementPoint'],
    ) -> list['DataPoint']:
        result = []
        for mp in pos:
            dp = await asyncio.to_thread(
                self.exp._simulate_datapoint,
                mp.base.model, mp.base.x, mp.base.t, mp.base.intent, None,
            )
            dp.movet = mp.base.movet or 0.0
            self._last_x = mp.base.x
            result.append(dp)
        return result
