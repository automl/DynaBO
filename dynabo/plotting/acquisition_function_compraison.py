"""
2D plot of hyperparameter λ (decay_beta) vs. acquisition function α.

Standard BO visualization style:
  - Top panel: true/surrogate function with sparse observations
  - Middle panel: acquisition function with argmax marked
  - Bottom panels: AF × prior combinations
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm


rng = np.random.default_rng(42)

lambda_values = np.linspace(0.05, 12.0, 300)


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


# --- True function (two modes) ---
# Scale factors derived from dense grid so evaluation is consistent at any x
_g1_scale = 0.45 / gaussian(lambda_values, mu=1.2, sigma=1.2).max()
_g2_scale = 0.60 / gaussian(lambda_values, mu=6.8, sigma=1.1).max()

def true_func_eval(x):
    return gaussian(x, mu=1.2, sigma=1.2) * _g1_scale + gaussian(x, mu=6.8, sigma=1.1) * _g2_scale


true_func = true_func_eval(lambda_values)

# --- Sparse observations (on the curve) ---
dot_lambda = np.array([0.2, 3.5, 5.2, 7.6, 9.5, 11.2])
dot_y = true_func_eval(dot_lambda)

# --- Acquisition function: Expected Improvement from GP ---
gp = GaussianProcessRegressor(
    kernel=Matern(),
    n_restarts_optimizer=5,
    normalize_y=True,
)
gp.fit(dot_lambda.reshape(-1, 1), dot_y)

mu_pred, sigma_pred = gp.predict(lambda_values.reshape(-1, 1), return_std=True)
f_best = dot_y.max()
Z = (mu_pred - f_best) / (sigma_pred + 1e-9)
acq_func = (mu_pred - f_best) * norm.cdf(Z) + sigma_pred * norm.pdf(Z)
acq_func = np.clip(acq_func, 0, None)
acq_func /= acq_func.max()
acq_argmax = lambda_values[np.argmax(acq_func)]

target_lambda = lambda_values[np.argmax(true_func)]

sigma_spiky = 12 / 9
sigma_uniform = 12 / 7

# One prior per local optimum, each with a slight offset
mu1_opt = 1.2 + 0.5  # slightly off the left local optimum
mu2_opt = 6.8 - 0.8  # slightly off the right local optimum (global)

prior1 = gaussian(lambda_values, mu=mu1_opt, sigma=sigma_uniform)
prior2 = gaussian(lambda_values, mu=mu2_opt, sigma=sigma_spiky)

# --- AF × Prior combinations ---
acq_prior1 = acq_func * prior1
acq_prior2 = acq_func * prior2
acq_prior_mul = acq_func * prior1 * prior2
acq_prior_sum = acq_func * (prior1 + prior2)

acq_prior1_argmax = lambda_values[np.argmax(acq_prior1)]
acq_prior2_argmax = lambda_values[np.argmax(acq_prior2)]
acq_prior_mul_argmax = lambda_values[np.argmax(acq_prior_mul)]
acq_prior_sum_argmax = lambda_values[np.argmax(acq_prior_sum)]

# --- Plot ---
fig, axes = plt.subplots(6, 1, figsize=(5, 11), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1, 1, 1, 1], "hspace": 0.08})
ax_top, ax_af, ax_p1, ax_p2, ax_mul, ax_sum = axes

# Top: true function + observations + priors
ax_top.plot(lambda_values, true_func, linewidth=1.5, color="steelblue", label=r"Target function: $f(\lambda)$")
ax_top.scatter(dot_lambda, dot_y, color="black", s=40, zorder=3, label="Observations")
ax_top.fill_between(lambda_values, prior1, alpha=0.25, color="mediumseagreen")
ax_top.plot(lambda_values, prior1, linewidth=1.2, color="mediumseagreen", linestyle="--", label=r"Prior: $\pi_1(\lambda)$")
ax_top.fill_between(lambda_values, prior2, alpha=0.25, color="darkorange")
ax_top.plot(lambda_values, prior2, linewidth=1.2, color="darkorange", linestyle="--", label=r"Prior: $\pi_2(\lambda)$")
ax_top.set_ylabel(r"$f(\lambda)$", fontsize=12)
ax_top.legend(fontsize=9, framealpha=0.8)
ax_top.grid(linestyle="--", linewidth=0.5, alpha=0.5)

# AF
ax_af.fill_between(lambda_values, acq_func, alpha=0.25, color="tomato")
ax_af.plot(lambda_values, acq_func, linewidth=1.5, color="tomato", label=r"$\alpha(\lambda)$")
ax_af.axvline(acq_argmax, color="tomato", linestyle="--", linewidth=1.0, alpha=0.7)
ax_af.set_ylabel(r"$\alpha(\lambda)$", fontsize=12)
ax_af.legend(fontsize=9, framealpha=0.8)
ax_af.grid(linestyle="--", linewidth=0.5, alpha=0.5)

# AF × π1
ax_p1.fill_between(lambda_values, acq_prior1, alpha=0.25, color="mediumseagreen")
ax_p1.plot(lambda_values, acq_prior1, linewidth=1.5, color="mediumseagreen", label=r"$\alpha \cdot \pi_1$")
ax_p1.axvline(acq_prior1_argmax, color="mediumseagreen", linestyle="--", linewidth=1.0, alpha=0.9)
ax_p1.set_ylabel(r"$\alpha \cdot \pi_1$", fontsize=12)
ax_p1.legend(fontsize=9, framealpha=0.8)
ax_p1.grid(linestyle="--", linewidth=0.5, alpha=0.5)

# AF × π2
ax_p2.fill_between(lambda_values, acq_prior2, alpha=0.25, color="darkorange")
ax_p2.plot(lambda_values, acq_prior2, linewidth=1.5, color="darkorange", label=r"$\alpha \cdot \pi_2$")
ax_p2.axvline(acq_prior2_argmax, color="darkorange", linestyle="--", linewidth=1.0, alpha=0.9)
ax_p2.set_ylabel(r"$\alpha \cdot \pi_2$", fontsize=12)
ax_p2.legend(fontsize=9, framealpha=0.8)
ax_p2.grid(linestyle="--", linewidth=0.5, alpha=0.5)

# AF × π1 × π2
ax_mul.fill_between(lambda_values, acq_prior_mul, alpha=0.25, color="mediumpurple")
ax_mul.plot(lambda_values, acq_prior_mul, linewidth=1.5, color="mediumpurple", label=r"$\alpha \cdot \pi_1 \cdot \pi_2$")
ax_mul.axvline(acq_prior_mul_argmax, color="mediumpurple", linestyle="--", linewidth=1.0, alpha=0.9)
ax_mul.set_ylabel(r"$\alpha \cdot \pi_1 \cdot \pi_2$", fontsize=12)
ax_mul.legend(fontsize=9, framealpha=0.8)
ax_mul.grid(linestyle="--", linewidth=0.5, alpha=0.5)

# AF × (π1 + π2)
ax_sum.fill_between(lambda_values, acq_prior_sum, alpha=0.25, color="steelblue")
ax_sum.plot(lambda_values, acq_prior_sum, linewidth=1.5, color="steelblue", label=r"$\alpha \cdot (\pi_1 + \pi_2)$")
ax_sum.axvline(acq_prior_sum_argmax, color="steelblue", linestyle="--", linewidth=1.0, alpha=0.9)
ax_sum.set_xlabel(r"$\lambda$", fontsize=12)
ax_sum.set_ylabel(r"$\alpha \cdot (\pi_1+\pi_2)$", fontsize=12)
ax_sum.legend(fontsize=9, framealpha=0.8)
ax_sum.grid(linestyle="--", linewidth=0.5, alpha=0.5)

plt.savefig("lambda_alpha_plot.pdf", bbox_inches="tight")
plt.show()
print("Saved lambda_alpha_plot.pdf")
