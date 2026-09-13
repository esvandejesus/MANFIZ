"""Native Python implementations of the supplied three 3-input/2-output plants."""
from dataclasses import dataclass
import numpy as np
from .data import matrix, NARXSpec

AMPLITUDES = np.array([[.80,.40,.20],[.70,.35,.20],[.60,.40,.25]])
FREQUENCIES = np.array([[.013,.031,.071],[.017,.043,.083],[.011,.037,.067]])
PHASE_STEPS = np.array([[.31,.53,.79],[.47,.71,1.07],[.23,.61,.97]])


@dataclass
class BenchmarkRun:
    u: np.ndarray
    y: np.ndarray
    y_true: np.ndarray
    phase: np.ndarray
    noise_seed: int

    def regression(self, spec=None, *, clean=False, group=0):
        spec = NARXSpec() if spec is None else spec
        return spec.transform(self.u, self.y_true if clean else self.y, group=group)


def simulate(system, u):
    """Deterministic, noise-free plant outputs; system is 1, 2, or 3."""
    u = matrix(u, 'u')
    if u.shape[1] != 3 or system not in (1,2,3):
        raise ValueError('Benchmarks require system in (1,2,3) and three inputs')
    n = len(u); y = np.zeros((n,2))
    if system in (1,2):
        for k in range(2,n):
            if system == 1:
                y[k,0] = (.55*y[k-1,0]-.12*y[k-2,0]+.08*y[k-1,1]+.18*np.sin(y[k-1,0])
                          +.30*u[k-1,0]+.18*u[k-2,1]-.15*u[k-1,2]+.06*u[k-1,0]*u[k-1,1])
                y[k,1] = (.50*y[k-1,1]-.10*y[k-2,1]+.08*y[k-1,0]+.15*np.tanh(y[k-1,1])
                          +.20*u[k-2,0]-.26*u[k-1,1]+.24*u[k-2,2]+.05*u[k-1,1]**2
                          +.04*u[k-1,0]*u[k-1,2])
            else:
                y[k,0] = (.62*y[k-1,0]-.16*y[k-2,0]+.10*y[k-1,1]+.28*np.tanh(1.4*u[k-1,0])
                          +.10*u[k-1,1]**3-.14*u[k-2,2]+.07*u[k-1,0]*u[k-1,2])
                y[k,1] = (.58*y[k-1,1]-.14*y[k-2,1]+.10*y[k-1,0]+.22*np.sin(1.3*u[k-1,1])
                          +.18*np.tanh(u[k-1,2])+.08*u[k-2,0]**2-.05*u[k-1,1]*u[k-1,2])
    else:
        x = np.zeros((n,3))
        for k in range(n-1):
            x1,x2,x3 = x[k]
            x[k+1] = [.72*x1+.10*x2+.18*np.tanh(u[k,0])+.10*u[k,1]*u[k,2],
                       -.08*x1+.80*x2+.15*np.sin(u[k,1])+.12*u[k,2],
                       .05*x1+.12*x2+.68*x3+.10*u[k,0]**2-.08*u[k,2]**2]
        y[:,0] = x[:,0]+.15*x[:,1]**2
        y[:,1] = x[:,1]+.20*np.tanh(x[:,2])+.05*x[:,0]*x[:,2]
    return y


def multisine(n_samples=1800, *, amplitude=1., phase=None):
    if int(n_samples) != n_samples or n_samples < 3 or not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError('Invalid length or amplitude')
    phase = np.zeros((3,3)) if phase is None else np.asarray(phase, dtype=float)
    if phase.shape != (3,3) or not np.all(np.isfinite(phase)):
        raise ValueError('phase must be a finite 3x3 matrix')
    k = np.arange(n_samples)[:,None,None]
    return amplitude*np.sum(AMPLITUDES*np.sin(2*np.pi*FREQUENCIES*k+phase), axis=2)


def make_run(system, *, n_samples=1800, amplitude=1., noise_std=.01,
             phase=None, phase_seed=0, noise_seed=1):
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError('noise_std must be finite and nonnegative')
    if phase is None:
        phase = np.random.default_rng(phase_seed).uniform(0,2*np.pi,(3,3))
    u = multisine(n_samples, amplitude=amplitude, phase=phase)
    clean = simulate(system,u)
    noise = np.sqrt(3)*noise_std*np.random.default_rng(noise_seed).uniform(-1,1,clean.shape)
    return BenchmarkRun(u,clean+noise,clean,np.asarray(phase).copy(),int(noise_seed))


def make_training_runs(system, *, n_samples=1800, seed=20260913):
    """Four training runs, one validation run and one held-out run.

    Original deterministic phase protocol; newly generated NumPy PCG64 noise.
    This intentionally does not claim bitwise MATLAB RNG equivalence.
    """
    return [make_run(system,n_samples=n_samples,amplitude=1.15 if r == 6 else 1.,
                     phase=np.mod(r*PHASE_STEPS,2*np.pi),noise_std=.01,
                     noise_seed=seed+1000*system+r) for r in range(1,7)]
