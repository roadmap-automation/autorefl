"""HTTP client for the AutoRefl fit server (autorefl.fit_server).

Replaces Refl1DClient usage in AutoReflBase: instead of serialising via the
bumps webview WebSocket protocol, FitClient sends the problem over HTTP and
receives HDF5 bytes containing both the DREAM state and Q-profiles.
"""

import asyncio
import io
import json
import logging
from typing import Callable, Dict, List, Optional, Tuple

import aiohttp
import h5py
import numpy as np
from bumps.serialize import serialize as bumps_serialize

logger = logging.getLogger(__name__)

# aiohttp default is 1 MB; fits can be large
_RESPONSE_LIMIT = 512 * 1024 * 1024  # 512 MB


def _encode_tdtldl(calc_tdtldl) -> str:
    """Serialise per-model (T,dT,L,dL) tuples to JSON."""
    return json.dumps([
        [arr.tolist() for arr in model_coords]
        for model_coords in calc_tdtldl
    ])


def _read_qprofs(f: h5py.File) -> List[np.ndarray]:
    nmodels = int(f.attrs['nmodels'])
    return [f[f'qprofs/model_{i}'][()] for i in range(nmodels)]


class FitClient:
    """Thin async HTTP client for the autorefl fit server.

    Parameters
    ----------
    host:
        Hostname or IP of the fit server.
    port:
        TCP port the fit server is listening on.
    max_retries:
        Number of reconnect attempts on ClientError before giving up.
    retry_delay_s:
        Seconds to wait between reconnect attempts.
    """

    def __init__(
        self,
        host: str = '127.0.0.1',
        port: int = 5031,
        max_retries: int = 3,
        retry_delay_s: float = 10.0,
    ) -> None:
        self.base_url = f'http://{host}:{port}'
        self._session: Optional[aiohttp.ClientSession] = None
        self._max_retries = max_retries
        self._retry_delay_s = retry_delay_s

    async def connect(self) -> None:
        connector = aiohttp.TCPConnector()
        self._session = aiohttp.ClientSession(
            connector=connector,
            connector_owner=True,
        )

    async def disconnect(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.disconnect()

    def _session_or_raise(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise RuntimeError('FitClient not connected — call connect() first.')
        return self._session

    async def _reconnect(self) -> None:
        try:
            if self._session is not None:
                await self._session.close()
        except Exception:
            pass
        self._session = None
        await self.connect()

    async def _with_retry(self, make_coro):
        """Run make_coro(), reconnecting and retrying on aiohttp.ClientError."""
        for attempt in range(self._max_retries + 1):
            try:
                return await make_coro()
            except aiohttp.ClientError as exc:
                if attempt == self._max_retries:
                    raise
                logger.warning(
                    'FitClient: connection error (%s), reconnecting in %.0fs (attempt %d/%d)',
                    exc, self._retry_delay_s, attempt + 1, self._max_retries,
                )
                await asyncio.sleep(self._retry_delay_s)
                await self._reconnect()

    async def post_fit(
        self,
        problem,
        fit_options: dict,
        calc_tdtldl,
        oversampling: int,
        resolution: str,
        warm_start: bool = False,
        requested_derived_labels: List[str] = [],
        timeout_s: float = 7200.0,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> Tuple[dict, List[np.ndarray]]:
        """Run DREAM fit + Q-profile calculation on the remote server.

        Returns
        -------
        fit_fields : dict
            Keys: ``chains`` (nsteps × nchains × npars), ``logp`` (nsteps × nchains),
            ``best_x`` (npars,), ``best_logp`` (float),
            ``draw_points`` (ndraw × npars), ``draw_logp`` (ndraw,),
            ``derived_draws`` (ndraw × D, only present if requested_derived_labels non-empty).
        qprofs : list[np.ndarray]
            One array per model, shape (ndraw, nQ).
        """
        problem_json = json.dumps(bumps_serialize(problem))
        encoded_tdtldl = _encode_tdtldl(calc_tdtldl)
        timeout = aiohttp.ClientTimeout(total=timeout_s)

        async def _impl():
            session = self._session_or_raise()
            data = aiohttp.FormData()
            data.add_field('problem', problem_json, content_type='application/json')
            data.add_field('fit_options', json.dumps(fit_options))
            data.add_field('warm_start', 'true' if warm_start else 'false')
            data.add_field('calc_tdtldl', encoded_tdtldl)
            data.add_field('oversampling', str(oversampling))
            data.add_field('resolution', resolution)
            if requested_derived_labels:
                data.add_field('requested_derived_labels', json.dumps(requested_derived_labels))
            async with session.post(
                f'{self.base_url}/fit',
                data=data,
                timeout=timeout,
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f'Fit server returned {resp.status}: {body}')
                # Read streaming response: JSON progress lines → __HDF5__ sentinel → binary
                while True:
                    line_bytes = await resp.content.readline()
                    if not line_bytes:
                        raise RuntimeError('Fit server stream ended without HDF5 sentinel')
                    line = line_bytes.rstrip(b'\n').decode('utf-8', errors='replace')
                    if line == '__HDF5__':
                        break
                    if progress_callback is not None:
                        try:
                            progress_callback(json.loads(line))
                        except (json.JSONDecodeError, KeyError):
                            pass
                return await resp.content.read(-1)

        hdf5_bytes = await self._with_retry(_impl)
        return _parse_fit_hdf5(hdf5_bytes)

    async def post_qprofiles(
        self,
        problem,
        draw_points: np.ndarray,
        calc_tdtldl,
        oversampling: int,
        resolution: str,
        timeout_s: float = 600.0,
    ) -> List[np.ndarray]:
        """Calculate Q-profiles for arbitrary draw points (no fitting).

        Used by ``AutoReflBase.initial_points()``.

        Returns
        -------
        qprofs : list[np.ndarray]
            One array per model, shape (npoints, nQ).
        """
        problem_json = json.dumps(bumps_serialize(problem))
        encoded_tdtldl = _encode_tdtldl(calc_tdtldl)
        encoded_points = json.dumps(draw_points.tolist())
        timeout = aiohttp.ClientTimeout(total=timeout_s)

        async def _impl():
            session = self._session_or_raise()
            data = aiohttp.FormData()
            data.add_field('problem', problem_json, content_type='application/json')
            data.add_field('draw_points', encoded_points)
            data.add_field('calc_tdtldl', encoded_tdtldl)
            data.add_field('oversampling', str(oversampling))
            data.add_field('resolution', resolution)
            async with session.post(
                f'{self.base_url}/qprofiles',
                data=data,
                timeout=timeout,
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f'Fit server /qprofiles returned {resp.status}: {body}')
                return await resp.read()

        hdf5_bytes = await self._with_retry(_impl)
        buf = io.BytesIO(hdf5_bytes)
        with h5py.File(buf, 'r') as f:
            return _read_qprofs(f)

    async def post_setup(
        self,
        problem,
        timeout_s: float = 300.0,
    ) -> dict:
        """Run one-time server setup: discover derived parameters and prior scales.

        Returns
        -------
        dict with keys:
            ``native_labels``   — list[str]
            ``derived_labels``  — list[str]
            ``prior_scales``    — list[float], one per derived label
            ``native_draws``    — np.ndarray (N, npars), prior draws that evaluated OK
            ``derived_draws``   — np.ndarray (N, D), derived values at those draws
                                  (shape (N, 0) when no derived parameters)
        """
        problem_json = json.dumps(bumps_serialize(problem))
        timeout = aiohttp.ClientTimeout(total=timeout_s)

        async def _impl():
            session = self._session_or_raise()
            data = aiohttp.FormData()
            data.add_field('problem', problem_json, content_type='application/json')
            async with session.post(
                f'{self.base_url}/setup',
                data=data,
                timeout=timeout,
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f'Fit server /setup returned {resp.status}: {body}')
                return await resp.read()

        hdf5_bytes = await self._with_retry(_impl)
        buf = io.BytesIO(hdf5_bytes)
        npars = len(problem.getp())
        with h5py.File(buf, 'r') as f:
            native_labels  = json.loads(f.attrs['native_labels'])
            derived_labels = json.loads(f.attrs['derived_labels'])
            prior_scales   = json.loads(f.attrs['prior_scales'])
            native_draws   = f['setup/native_draws'][()] if 'setup/native_draws' in f \
                             else np.empty((0, npars))
            derived_draws  = f['setup/derived_draws'][()] if 'setup/derived_draws' in f \
                             else np.empty((0, len(derived_labels)))

        return {
            'native_labels':  native_labels,
            'derived_labels': derived_labels,
            'prior_scales':   prior_scales,
            'native_draws':   native_draws,
            'derived_draws':  derived_draws,
        }

    async def post_reset(self, timeout_s: float = 10.0) -> None:
        """Clear the server warm-start chain and derived parameter state."""
        timeout = aiohttp.ClientTimeout(total=timeout_s)

        async def _impl():
            session = self._session_or_raise()
            async with session.post(
                f'{self.base_url}/reset',
                timeout=timeout,
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f'Fit server /reset returned {resp.status}: {body}')

        await self._with_retry(_impl)

    async def post_derived_draws(
        self,
        problem,
        draw_points: np.ndarray,
        requested_labels: List[str],
        timeout_s: float = 600.0,
    ) -> np.ndarray:
        """Evaluate molgroups derived quantities at posterior draw points.

        Parameters
        ----------
        problem:
            The bumps FitProblem (dill-serialised for transport).
        draw_points:
            N × P array of posterior draw points from the last fit.
        requested_labels:
            Subset of derived labels (from post_setup) to compute and return.

        Returns
        -------
        derived_draws : np.ndarray, shape (N, D)
            One column per requested label.
        """
        problem_json = json.dumps(bumps_serialize(problem))
        encoded_points = json.dumps(draw_points.tolist())
        timeout = aiohttp.ClientTimeout(total=timeout_s)

        async def _impl():
            session = self._session_or_raise()
            data = aiohttp.FormData()
            data.add_field('problem', problem_json, content_type='application/json')
            data.add_field('draw_points', encoded_points)
            data.add_field('requested_labels', json.dumps(requested_labels))
            async with session.post(
                f'{self.base_url}/derived_draws',
                data=data,
                timeout=timeout,
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f'Fit server /derived_draws returned {resp.status}: {body}')
                return await resp.read()

        hdf5_bytes = await self._with_retry(_impl)
        buf = io.BytesIO(hdf5_bytes)
        with h5py.File(buf, 'r') as f:
            return f['derived/draws'][()]

    async def post_stop_fit(self, timeout_s: float = 5.0) -> None:
        """Signal the fit server to stop the current DreamFit run early."""
        timeout = aiohttp.ClientTimeout(total=timeout_s)

        async def _impl():
            session = self._session_or_raise()
            async with session.post(
                f'{self.base_url}/stop_fit',
                timeout=timeout,
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f'Fit server /stop_fit returned {resp.status}: {body}')

        await self._with_retry(_impl)

    async def is_alive(self) -> bool:
        session = self._session_or_raise()
        try:
            async with session.get(f'{self.base_url}/status', timeout=aiohttp.ClientTimeout(total=5)) as resp:
                return resp.status == 200
        except Exception:
            return False


def _parse_fit_hdf5(hdf5_bytes: bytes) -> Tuple[dict, List[np.ndarray]]:
    """Deserialise the HDF5 response from /fit into Python objects."""
    buf = io.BytesIO(hdf5_bytes)
    with h5py.File(buf, 'r') as f:
        fit_fields = {
            'chains':       f['chains/points'][()],
            'logp':         f['chains/logp'][()],
            'best_x':       f['best/x'][()],
            'best_logp':    float(f.attrs['best_logp']),
            'draw_points':  f['draw/points'][()],
            'draw_logp':    f['draw/logp'][()],
        }
        if 'derived/draws' in f:
            fit_fields['derived_draws'] = f['derived/draws'][()]
        qprofs = _read_qprofs(f)
    return fit_fields, qprofs
