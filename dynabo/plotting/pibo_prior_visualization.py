"""
4-panel illustration of how a helpful prior works in PiBO.

  Panel 1: Target function f(λ) + sparse observations
  Panel 2: f(λ) + observations + GP surrogate belief (mean ± 95 % CI)
  Panel 3: Helpful prior π(λ)
  Panel 4: Adapted acquisition α(λ) · π(λ) with argmax marked
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm


rng = np.random.default_rng(42)

x = np.linspace(0.0, 10.0, 400)

# ---------------------------------------------------------------------------
# Target function  (two modes — right peak is global optimum)
# ---------------------------------------------------------------------------

def _gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def target(x):
    return 0.45 * _gauss(x, mu=2.2, sigma=0.9) + 0.85 * _gauss(x, mu=7.3, sigma=1.1)


f = target(x)

# ---------------------------------------------------------------------------
# Sparse observations — sampled from the curve with tiny noise.
# The point near x=4 is deliberately elevated above the true function so
# the surrogate peaks there, making the plain acquisition point left of the
# true global optimum.  The global-peak region (x≈7.3) is left unexplored.
# ---------------------------------------------------------------------------
# All observations sit on the curve.  The left peak (x≈2.2) is well sampled
# so it dominates f_best; the global peak (x≈7.3) is left unexplored so the
# plain acquisition focuses left — the prior then redirects to the right.
obs_x = np.array([0.5, 1.5, 3.8, 5.2, 9.2])
obs_y = target(obs_x) + rng.normal(0, 0.015, size=len(obs_x))

# ---------------------------------------------------------------------------
# GP surrogate — fix length_scale so it captures the peak widths (~1 unit)
# rather than letting the optimiser pick an over-smoothed scale.
# ---------------------------------------------------------------------------
gp = GaussianProcessRegressor(
    kernel=Matern(length_scale=0.8, length_scale_bounds="fixed", nu=2.5),
    n_restarts_optimizer=10,
    normalize_y=True,
    alpha=1e-4,
)
gp.fit(obs_x.reshape(-1, 1), obs_y)

mu, sigma = gp.predict(x.reshape(-1, 1), return_std=True)
ci_lo = mu - 1.96 * sigma
ci_hi = mu + 1.96 * sigma

# ---------------------------------------------------------------------------
# Acquisition function: Expected Improvement
# ---------------------------------------------------------------------------
f_best = obs_y.max()
with np.errstate(divide="ignore"):
    Z = (mu - f_best) / (sigma + 1e-9)
acq = (mu - f_best) * norm.cdf(Z) + sigma * norm.pdf(Z)
acq = np.clip(acq, 0, None)
acq_norm = acq / (acq.max() + 1e-12)

# ---------------------------------------------------------------------------
# Helpful prior — peaked near the true global optimum (expert knowledge)
# ---------------------------------------------------------------------------
prior = _gauss(x, mu=7.0, sigma=2.0)   # centred slightly off the true peak

# ---------------------------------------------------------------------------
# Adapted acquisition α · π
# ---------------------------------------------------------------------------
acq_adapted = acq_norm * prior
acq_adapted_argmax = x[np.argmax(acq_adapted)]
acq_argmax = x[np.argmax(acq_norm)]

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
C_TRUE   = "#2E86C1"        # steelblue  — target function
C_SURRG  = "#2E86C1"        # same hue for GP mean
C_CI     = "#AED6F1"        # light blue — CI band
C_OBS    = "#1a1a1a"        # near-black — observations
C_PRIOR  = "#27AE60"        # green      — helpful prior
C_ACQ    = "#E74C3C"        # red        — acquisition function
C_ADAPT  = "#8E44AD"        # purple     — adapted acquisition

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(
    4, 1,
    figsize=(5.5, 11),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 2, 1.5, 1.5], "hspace": 0.06},
)
ax_f, ax_gp, ax_pi, ax_ad = axes


def _grid(ax):
    ax.grid(linestyle="--", linewidth=0.5, alpha=0.45)
    ax.set_xlim(x[0], x[-1])


# ── Panel 1: target function + observations ──────────────────────────────────
ax_f.plot(x, f, color=C_TRUE, linewidth=1.8, linestyle="--", label=r"Target $f(\lambda)$")
ax_f.scatter(obs_x, obs_y, color=C_OBS, s=45, zorder=4, label="Observations")
ax_f.set_ylabel(r"$f(\lambda)$", fontsize=12)
ax_f.legend(fontsize=9, framealpha=0.85, loc="upper left")
_grid(ax_f)

# ── Panel 2: target + observations + GP surrogate ───────────────────────────
ax_gp.fill_between(x, ci_lo, ci_hi, color=C_CI, alpha=0.55, label="95 % CI")
ax_gp.plot(x, mu, color=C_SURRG, linewidth=1.8, label=r"Surrogate $\hat{f}(\lambda)$")
ax_gp.plot(x, f, color=C_TRUE, linewidth=1.2, linestyle="--", alpha=0.45, label=r"True $f(\lambda)$")
ax_gp.scatter(obs_x, obs_y, color=C_OBS, s=45, zorder=4, label="Observations")
ax_gp.set_ylabel(r"$\hat{f}(\lambda)$", fontsize=12)
ax_gp.legend(fontsize=9, framealpha=0.85, loc="upper left", ncol=2)
_grid(ax_gp)

# ── Panel 3: helpful prior ───────────────────────────────────────────────────
ax_pi.fill_between(x, prior, alpha=0.25, color=C_PRIOR)
ax_pi.plot(x, prior, color=C_PRIOR, linewidth=1.8, label=r"Prior $\pi(\lambda)$")
ax_pi.set_ylabel(r"$\pi(\lambda)$", fontsize=12)
ax_pi.legend(fontsize=9, framealpha=0.85, loc="upper left")
_grid(ax_pi)

# ── Panel 4: adapted acquisition α · π ──────────────────────────────────────
# Show plain α in faint red for reference
ax_ad.fill_between(x, acq_norm, alpha=0.12, color=C_ACQ)
ax_ad.plot(x, acq_norm, color=C_ACQ, linewidth=1.2, linestyle="--", alpha=0.6,
           label=r"$\alpha(\lambda)$")
# Adapted (normalised to same scale for readability)
ad_norm = acq_adapted / (acq_adapted.max() + 1e-12)
ax_ad.fill_between(x, ad_norm, alpha=0.25, color=C_ADAPT)
ax_ad.plot(x, ad_norm, color=C_ADAPT, linewidth=1.8,
           label=r"$\alpha(\lambda)\cdot\pi(\lambda)$")
ax_ad.axvline(acq_argmax, color=C_ACQ, linestyle=":", linewidth=1.4, alpha=0.9)
ax_ad.axvline(acq_adapted_argmax, color=C_ADAPT, linestyle=":", linewidth=1.4, alpha=0.9)
ax_ad.set_ylabel(r"$\alpha \cdot \pi$", fontsize=12)
ax_ad.set_xlabel(r"$\lambda$", fontsize=12)
ax_ad.legend(fontsize=9, framealpha=0.85, loc="upper left")
_grid(ax_ad)

# Shared y-lower bound at 0 for non-GP panels
for ax in [ax_f, ax_pi, ax_ad]:
    ax.set_ylim(bottom=0)

plt.savefig("pibo_prior_visualization.png", bbox_inches="tight", dpi=600)
plt.show()
print("Saved pibo_prior_visualization.png")
