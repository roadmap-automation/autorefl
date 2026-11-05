import numpy as np
from scipy.stats import poisson, norm
from refl1d.probe.probe import PolarizedNeutronProbe


def _iter_subprobes(probe):
    """Return individual probes from a regular or PolarizedNeutronProbe."""
    if isinstance(probe, PolarizedNeutronProbe):
        return [xs for xs in probe.xs if xs is not None]
    return [probe]

# numpy's Poisson RNG overflows above ~2^31; Gaussian is indistinguishable from
# Poisson at high counts anyway, so switch well below that limit.
_POISSON_MAX_LAM = 1e8

def _rvs(lam):
    """Poisson sample, falling back to Gaussian for large lambda."""
    lam = np.asarray(lam, dtype=np.float64)
    if np.any(lam > _POISSON_MAX_LAM):
        return norm.rvs(loc=lam, scale=np.sqrt(np.maximum(lam, 0.0)), size=lam.shape)
    return poisson.rvs(lam, size=lam.shape)


def sim_data_N(R, incident_neutrons: np.ndarray, resid_bkg=0.0, meas_bkg=0.0):
    R = np.array(R, ndmin=1)
    _bR = np.ones_like(R, dtype=np.float64) * (meas_bkg - resid_bkg) * incident_neutrons
    _R = (R + meas_bkg - resid_bkg) * np.asarray(incident_neutrons, dtype=np.float64)
    _Ninc = np.asarray(incident_neutrons, dtype=np.float64)

    N    = _rvs(_R)
    bNp  = _rvs(_bR)
    bNm  = _rvs(_bR)
    Ninc = _rvs(_Ninc)

    return N, (bNp, bNm), Ninc

_FWHM = 2 * np.sqrt(2 * np.log(2))

def calc_expected_R(fitness, T, dT, L, dL, oversampling=None, resolution='normal'):
    """Calculate expected reflectivity for one model at given T/L arrays.

    Always returns a list of R arrays: one element for an unpolarized probe,
    one per active cross-section for a PolarizedNeutronProbe.  Callers index
    with [0] for the unpolarized case or extend/iterate for the polarized case.

    dT and dL arrive in 1-sigma (reductus convention); refl1d _set_TLR expects FWHM.
    oversampling is accepted for API compatibility but ignored: calc_Q is driven by
    probe.oversampled_regions (set in AutoReflBase.__init__).
    """
    probe = fitness.probe
    if isinstance(probe, PolarizedNeutronProbe):
        # _set_TLR and _apply_oversamplings live on the xs sub-probes, not the parent.
        # All xs share the same T/L (same measurement angle); set them all then read
        # each xs's reflectivity out of experiment.reflectivity().
        active_xs_indices = [i for i, xs in enumerate(probe.xs) if xs is not None]
        for xs in [probe.xs[i] for i in active_xs_indices]:
            xs._set_TLR(T, dT * _FWHM, L, dL * _FWHM, R=None, dR=None, dQ=None)
            xs.resolution = resolution
            xs._apply_oversamplings()
        # _QdQ_key() only tracks theta_offset/sample_broadening, not T/L directly.
        # Invalidate the union cache so _calculate_union recomputes from the new T/L.
        probe._union_cache_key = None
        fitness.update()
        all_R = fitness.reflectivity()
        # apply_beam on PolarizedNeutronProbe returns [(Q, R), ...]; extract R array only.
        return [all_R[i][1] for i in active_xs_indices]
    probe._set_TLR(T, dT * _FWHM, L, dL * _FWHM, R=None, dR=None, dQ=None)
    probe.resolution = resolution
    probe._apply_oversamplings()
    fitness.update()
    return [fitness.reflectivity()[1]]
