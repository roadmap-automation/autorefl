import asyncio
import datetime
import copy
import os
import json
import time
import traceback

import numpy as np

from autorefl.remote.nicedata import Signaller
from autorefl.remote.monitor import SocketServer, buttonhandler, emit_history, emit_clear
from autorefl.backend import MeasurementBackend

from autorefl.autorefl import AutoReflExperiment
from autorefl.calibration import calibrate_intensity


class AutoReflLauncher:
    """Autonomous reflectometry optimization loop (asyncio-native).

    Lifecycle::

        from autorefl.remote.nicedata import NICECampaignTask, NICEBackend

        task = NICECampaignTask(instr.trajectoryMotors(), filename)
        launcher = AutoReflLauncher(exp, client, signals, maxtime, pathname,
                                    backend=NICEBackend(task))
        await asyncio.gather(
            asyncio.to_thread(api.serve_tasks, task),
            launcher.run(),
        )

    For simulation, use :class:`~autorefl.backend.SimulatedBackend`.
    For custom hardware, subclass :class:`~autorefl.backend.MeasurementBackend`
    and implement ``measure()``.
    """

    def __init__(self,
                 exp: AutoReflExperiment,
                 client,
                 signals: Signaller,
                 maxtime: float,
                 pathname: str,
                 backend: MeasurementBackend,
                 intensity_template=None,
                 intensity_files=None,
                 reduction_template=None,
                 q_rebin=None,
                 min_steps: int = 0,
                 max_chi2: float | None = None,
                 param_uncertainty: dict | None = None) -> None:
        if backend is None:
            raise ValueError(
                'backend is required. Pass a MeasurementBackend instance '
                '(NICEBackend, SimulatedBackend, or a custom subclass).'
            )
        self.exp = exp
        self.client = client
        self.signals = signals
        self.maxtime = maxtime
        self.pathname = pathname
        self.backend = backend
        self.intensity_template = intensity_template
        self.intensity_files = intensity_files
        # reduction_template: same reductus template JSON used for real-time reduction
        # via reductus after each measurement step (hardware mode only). If None, the
        # DataPoint/simulation path is used for the fitter.
        self.reduction_template = reduction_template
        self.q_rebin = q_rebin
        self.min_steps = min_steps
        self.max_chi2 = max_chi2
        self.param_uncertainty = param_uncertainty or {}
        self._stop = asyncio.Event()

    def _check_convergence_criteria(self, step, k: int) -> bool:
        """Return True if any early-stop criterion is met after min_steps."""
        if self.param_uncertainty and step.draw_pts is not None:
            labels = list(self.exp.problem.labels()) + list(self.exp._requested_derived_labels)
            met = []
            for param_name, threshold in self.param_uncertainty.items():
                try:
                    idx = labels.index(param_name)
                except ValueError:
                    print(f'AutoLauncher: uncertainty criterion param {param_name!r} not found in labels — skipping', flush=True)
                    continue
                if idx >= step.draw_pts.shape[1]:
                    continue
                sigma = float(np.std(step.draw_pts[:, idx]))
                ratio = sigma / threshold
                print(f'AutoLauncher: {param_name} uncertainty = {sigma:.4g} (target < {threshold:.4g}, {ratio:.2f}x)', flush=True)
                met.append(sigma < threshold)
            if met and all(met) and k >= self.min_steps:
                print('AutoLauncher: all uncertainty criteria met — stopping', flush=True)
                return True
        if k < self.min_steps:
            return False
        if self.max_chi2 is not None and step.final_chisq_value is not None:
            if step.final_chisq_value > self.max_chi2:
                print(f'AutoLauncher: chi-sq {step.final_chisq_value:.3f} > max_chi2 {self.max_chi2} — stopping', flush=True)
                return True
        return False

    def stop(self) -> None:
        self._stop.set()
        self.backend.stop()
        try:
            asyncio.create_task(self.client.post_stop_fit())
        except RuntimeError:
            pass
        self.signals.global_start.set()
        self.signals.measurement_queue_updated.set()
        self.signals.first_measurement_complete.set()

    def _save_checkpoint(self, step_k: int) -> None:
        """Write collected DataPoints for step_k to a JSON file in the campaign directory.

        Each record contains x, t, intent, per-channel N (counts) and Ninc
        (calibrated incident flux × time from instrument.intensity) so the
        data can be verified independently before it enters the fitter.
        """
        from autorefl.datastruct import data_attributes
        step = self.exp.steps[step_k]
        records = []
        for pt in step.points:
            record = {
                'x': float(pt.x),
                't': float(pt.t),
                'model': int(pt.model),
                'intent': str(pt.intent),
            }
            for attr in data_attributes:
                record[attr] = np.array(getattr(pt, attr)).ravel().tolist()
            records.append(record)
        path = os.path.join(self.pathname, f'checkpoint_step_{step_k:04d}.json')
        with open(path, 'w') as f:
            json.dump({'step': step_k, 'n_points': len(records), 'points': records}, f, indent=2)
        print(f'AutoLauncher: checkpoint saved — step {step_k}, {len(records)} point(s) → {path}', flush=True)

    async def _run_reductus_reduction(self, step_k: int) -> None:
        """Run reductus reduction on the current campaign snapshot and update exp._current_reduced.

        Only active when ``reduction_template`` and ``intensity_files`` are set and the
        backend provides a campaign snapshot (NICEBackend).  No-op in simulation mode.
        """
        if not self.reduction_template or not self.intensity_files:
            return
        file_info = getattr(self.backend, 'campaign_file_info', None)
        if file_info is None:
            return
        from autorefl.reduction import reduce_autorefl_step
        try:
            print(f'AutoLauncher: running reductus reduction for step {step_k}', flush=True)
            reduced = await asyncio.to_thread(
                reduce_autorefl_step,
                self.reduction_template,
                file_info['path'],
                self.intensity_files,
                self.exp.curmodel,
                self.q_rebin,
            )
            new_reduced = list(self.exp._current_reduced or self.exp._init_reduced)
            new_reduced[self.exp.curmodel] = reduced
            self.exp._current_reduced = new_reduced
            print(f'AutoLauncher: reductus reduction complete for step {step_k}', flush=True)
        except Exception:
            import traceback
            print('AutoLauncher: reductus reduction failed — falling back to DataPoint path:', flush=True)
            traceback.print_exc()

    async def run(self) -> None:
        def _log_socket_error(task):
            if not task.cancelled() and task.exception() is not None:
                import traceback
                print('AutoLauncher: monitor failed:', ''.join(traceback.format_exception(task.exception())))

        socket_task = asyncio.create_task(SocketServer().serve())
        socket_task.add_done_callback(_log_socket_error)

        # register stop/start button callbacks
        buttonhandler.start_callbacks.append(self.signals.global_start.set)
        buttonhandler.stop_callbacks.append(self.stop)
        buttonhandler.terminate_count_callbacks.append(self.backend.terminate_count)
        def _terminate_fit() -> None:
            asyncio.create_task(self.client.post_stop_fit())
        buttonhandler.terminate_fit_callbacks.append(_terminate_fit)

        await asyncio.sleep(1)
        await self.signals.global_start.wait()

        if self.intensity_template is not None and self.intensity_files:
            print('AutoLauncher: calibrating intensity from scan files')
            await asyncio.to_thread(
                calibrate_intensity, self.exp.instrument, self.intensity_template, self.intensity_files
            )
            instr = self.exp.instrument
            if (instr.fixed_slits is None
                    and len(instr.s1_intens_calib) == 1
                    and np.isfinite(instr.sample_width)):
                s1 = float(instr.s1_intens_calib[0])
                s2 = s1 / instr._R12
                s3 = (s1 + s2) * (instr._L2S + instr._LS3) / instr._L12 + s2 + instr._S3Offset
                s4 = (s1 + s2) * (instr._L2S + instr._LS3 + instr._L34) / instr._L12 + s2 + instr._S3Offset
                instr.fixed_slits = (s1, s2, s3, s4)
                print(f'AutoLauncher: single-point calibration detected — flood beam mode, fixed_slits={instr.fixed_slits}')
        else:
            print('AutoLauncher: no intensity calibration provided, using simulation fallback')

        if not self.exp._ready:
            print('AutoLauncher: running setup to discover derived parameters')
            await self.exp.setup(self.client)

        print('AutoLauncher: calculating initial points')
        points, init_qprofs, init_step = await self.exp.initial_points(self.client)

        await emit_clear()

        try:
            await emit_history('autorefl_profiles', self.update_plot_profiles(init_qprofs))
            if init_step.all_foms is not None:
                await emit_history('autorefl_fom', self._fom_payload(init_step.all_foms))
        except Exception:
            import traceback
            traceback.print_exc()

        total_t = 0.0
        warm_start = False

        def _fit_progress(msg: dict) -> None:
            step = msg.get('step', '?')
            parts = [f'step {step}', f'best={msg["chisq"]}']
            if 'p' in msg:
                parts.append(f'p={msg["p"]:.3f}')
            if 'chi_median' in msg:
                parts.append(f'post={msg["chi_median"]}[{msg["chi_lo"]},{msg["chi_hi"]}]')
            asyncio.create_task(emit_history('fit_update', '  '.join(parts)))

        pending = list(points)
        self.exp.add_step([])
        batch_total = len(pending)
        batch_received = 0
        fit_task: asyncio.Task | None = None
        step_k = 0

        try:
            while total_t < self.maxtime and not self._stop.is_set():
                if not pending:
                    break

                if batch_received == 0:
                    print(f'AutoLauncher: Step {step_k}, total time so far: {total_t:.1f} s', flush=True)
                    await emit_history('new_step', {'step': step_k, 'text': f'Step {step_k}'})
                    def _q_range(x):
                        q = np.asarray(self.exp.instrument.x2q(x)).ravel()
                        return [float(q.min()), float(q.max())]
                    queue_data = [
                        {'step_id': mp.step_id, 'point_id': mp.point_id, 'point_index': i,
                         'model': mp.base.model, 'x': f'{mp.base.x:.4f}',
                         'q_range': _q_range(mp.base.x),
                         'intent': str(mp.base.intent), 'time': f'{mp.base.t:.1f}'}
                        for ptlist in pending for i, mp in enumerate(ptlist)
                    ]
                    await emit_history('queue_update', queue_data)

                pos = pending.pop(0)
                pos_data = await self.backend.measure_position(pos)

                step_id = pos[0].step_id
                if step_id < len(self.exp.steps):
                    self.exp.steps[step_id].points.extend(pos_data)
                batch_received += 1

                await self._run_reductus_reduction(step_k)

                try:
                    await emit_history('autorefl_data', self.update_plot_data(mark_new_step=(batch_received == 1)))
                except Exception:
                    traceback.print_exc()

                if batch_received == 1:
                    print('AutoLauncher: starting fit task', flush=True)
                    fit_task = asyncio.create_task(
                        self.exp.fit_step(self.client, warm_start=warm_start, progress_callback=_fit_progress)
                    )

                batch_done = (batch_received == batch_total)
                early_done = (fit_task is not None and fit_task.done() and not batch_done)

                if early_done or batch_done:
                    if fit_task is not None and not fit_task.done():
                        await self.client.post_stop_fit()
                    if fit_task is not None:
                        try:
                            await fit_task
                        except Exception:
                            traceback.print_exc()
                    fit_task = None

                    if self.backend.last_x is not None:
                        self.exp.instrument.x = self.backend.last_x

                    self._save_checkpoint(step_k)

                    step = self.exp.steps[step_k]
                    print('AutoLauncher: final chi-squared:', step.final_chisq, flush=True)

                    try:
                        if step.qprofs is not None:
                            await emit_history('autorefl_profiles', self.update_plot_profiles(step.qprofs))
                        await emit_history('autorefl_data', self.update_plot_data())
                        convergence = [
                            {'step': i, 'dH': s.dH, 'dH_marg': s.dH_marg, 'chisq': s.final_chisq_value}
                            for i, s in enumerate(self.exp.steps) if s.dH is not None
                        ]
                        await emit_history('autorefl_convergence', json.dumps(convergence))
                    except Exception:
                        traceback.print_exc()

                    if early_done:
                        pending.clear()

                    print('AutoLauncher: calculating FOM', flush=True)
                    new_points = await asyncio.to_thread(self.exp.take_step, allow_repeat=False)

                    if early_done:
                        measured_xs = {dp.x for dp in self.exp.steps[step_k].points}
                        new_points = [p for p in new_points if p[0].base.x not in measured_xs]

                    if self._check_convergence_criteria(step, step_k):
                        await emit_history('autorefl_stop_reason', json.dumps({'reason': 'convergence_criterion', 'step': step_k}))
                        print('AutoLauncher: saving before early stop', flush=True)
                        await asyncio.to_thread(self.exp.save, self.pathname + '/autoexp0.pickle')
                        break

                    try:
                        if step.all_foms is not None:
                            await emit_history('autorefl_fom', self._fom_payload(step.all_foms))
                    except Exception:
                        traceback.print_exc()

                    print('AutoLauncher: saving', flush=True)
                    await asyncio.to_thread(self.exp.save, self.pathname + '/autoexp0.pickle')

                    self.exp.add_step([])
                    pending = list(new_points)
                    batch_received = 0
                    batch_total = len(pending)
                    warm_start = True
                    total_t = sum(pt.t + pt.movet for step in self.exp.steps for pt in step.points)
                    step_k += 1

        except Exception:
            print('AutoLauncher: unhandled exception in run loop:', flush=True)
            traceback.print_exc()
            raise
        finally:
            if fit_task is not None and not fit_task.done():
                try:
                    await fit_task
                except Exception:
                    pass
            self.backend.stop()
            socket_task.cancel()
            try:
                await socket_task
            except asyncio.CancelledError:
                pass

    def _fom_payload(self, all_foms) -> str:
        """Serialise all forecast-step FOMs for the autorefl_fom event.

        Returns a JSON list-of-lists: outer index = forecast step (0 = next
        measurement), inner index = model.  Each entry has model, x, fom, step.
        """
        return json.dumps([
            [
                {'model': m, 'x': list(self.exp.x[m]), 'fom': list(fom.tolist()), 'step': s}
                for m, fom in enumerate(step_foms)
            ]
            for s, step_foms in enumerate(all_foms)
        ])

    def update_plot_data(self, mark_new_step: bool = False) -> str:
        from autorefl.reduction import reduce, DataPoint2ReflData
        from autorefl.datastruct import Intent

        # Reductus path: use the already-reduced R(Q) from the template directly.
        reduced = self.exp._current_reduced or (
            self.exp._init_reduced if any(r is not None for r in self.exp._init_reduced) else None
        )
        if reduced is not None and any(r is not None for r in reduced):
            plotdata = {'data': [], 'new_data': []}
            for i, r in enumerate(reduced):
                if r is not None and r.spec is not None:
                    refl = r.spec
                    plotdata['data'].append({
                        'model': i,
                        'x': list(np.asarray(refl.Qz).ravel()),
                        'v': list(np.asarray(refl.v).ravel()),
                        'dv': list(np.asarray(refl.dv).ravel()),
                    })
            return json.dumps(plotdata)

        # DataPoint path (simulation mode): reduce from raw points.
        steps = self.exp.steps
        if mark_new_step and steps:
            old_steps, new_steps = steps[:-1], steps[-1:]
        else:
            old_steps, new_steps = steps, []

        plotdata = {'data': [], 'new_data': []}
        for i, mQ in enumerate(self.exp.measQ):
            def _reduce(step_list, _i=i, _mQ=mQ):
                spec = DataPoint2ReflData(_mQ, [pt for s in step_list for pt in s.points if pt.model == _i and pt.intent == Intent.spec])
                bkgp = DataPoint2ReflData(_mQ, [pt for s in step_list for pt in s.points if pt.model == _i and pt.intent == Intent.backp])
                bkgm = DataPoint2ReflData(_mQ, [pt for s in step_list for pt in s.points if pt.model == _i and pt.intent == Intent.backm])
                return reduce(spec, bkgp, bkgm)

            refl = _reduce(old_steps)
            if refl is not None:
                plotdata['data'].append({'model': i, 'x': list(refl.x), 'v': list(refl.v), 'dv': list(refl.dv)})

            if new_steps:
                refl_new = _reduce(new_steps)
                if refl_new is not None:
                    plotdata['new_data'].append({'model': i, 'x': list(refl_new.x), 'v': list(refl_new.v), 'dv': list(refl_new.dv)})

        return json.dumps(plotdata)

    def update_plot_profiles(self, allqprofs) -> str:
        from bumps.plotutil import form_quantiles
        plotdata = {'ci': []}
        for q, qprofs in zip(self.exp.measQ, allqprofs):
            _, ci = form_quantiles(qprofs, [68, 95])
            plotdata['ci'].append({
                'x': list(q),
                '68': {'lower': list(ci[0][0]), 'upper': list(ci[0][1])},
                '95': {'lower': list(ci[1][0]), 'upper': list(ci[1][1])},
            })
        return json.dumps(plotdata)

if __name__ == '__main__':

    # python -m autorefl.launch
    import os
    from autorefl.instrument import MAGIK, CANDOR
    from bumps.cli import load_model
    from autorefl.fit_client import FitClient
    from autorefl.remote.nicedata import NICECampaignTask, NICEBackend
    from nice.remote import connect as nice_connect

    instr = MAGIK()

    modelfile = 'example_model/ssblm_d2o.py'
    model = load_model(modelfile)

    bestpars = 'example_model/ssblm_d2o_tosb0.par'
    bestp = np.array([float(line.split(' ')[-1]) for line in open(bestpars, 'r').readlines()]) if bestpars is not None else None

    sel = [10, 11, 12, 13, 14]

    qstep_max = 0.0024
    qmax = 0.25
    qmin = 0.008
    qstep = 0.0005
    dq = np.linspace(qstep, qstep_max, int(np.ceil(2 * (qmax - qmin) / (qstep_max + qstep))))
    measQ = (qmin - qstep) + np.cumsum(dq)

    exp = AutoReflExperiment(model, measQ, instr,
                             bestpars=bestp,
                             meas_bkg=3e-6,
                             eta=0.5,
                             npoints=6,
                             select_pars=sel,
                             min_meas_time=20.0,
                             oversampling=5,
                             fit_options={'burn': 1000, 'steps': 100, 'pop': 8})
    if instr.name == 'MAGIK':
        exp.x = exp.measQ
    # CANDOR: __init__ default (linspace of angles) is sufficient

    fprefix = '%s_eta%0.2f_npoints%i' % (instr.name, exp.eta, exp.npoints)
    fn = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    pathname = fprefix + '_' + fn + '_testauto'
    os.makedirs(pathname, exist_ok=True)

    NICE_HOST = os.environ.get('NICE_HOST', 'localhost')
    NICE_CLIENT = os.environ.get('NICE_CLIENT', 'AutoRefl')

    async def main():
        signals = Signaller()
        client = FitClient(
            host=os.environ.get('AUTOREFL_FIT_HOST', '127.0.0.1'),
            port=int(os.environ.get('AUTOREFL_FIT_PORT', '5031')),
        )

        nice_api = await asyncio.to_thread(nice_connect, NICE_HOST, NICE_CLIENT)
        nice_task = NICECampaignTask(
            motors_to_move=instr.trajectoryMotors(),
            filename=fprefix,
        )

        await client.connect()
        try:
            launcher = AutoReflLauncher(exp, client, signals, maxtime=7200, pathname=pathname,
                                        backend=NICEBackend(nice_task))
            await asyncio.gather(
                asyncio.to_thread(nice_api.serve_tasks, nice_task),
                launcher.run(),
            )
        finally:
            await client.disconnect()
            nice_api.end_serve()
            nice_api.disconnect()

    print("launching")
    asyncio.run(main())
