"""
pgm.py
======
Probabilistic graphical model (simple naive-Bayes style) for cross-feeding.

Latent binary states per (chain, metabolite):

    Z_prodA  = 1  iff  strain A (solo step)     produces the metabolite
    Z_consB  = 1  iff  strain B (partner step)  consumes the metabolite
    Z_prodB  = 1  iff  strain B (mirror solo)   produces it independently

Observed log2FCs at each step are assumed Gaussian around either:
    * MU_ACTIVE (produced or consumed, depending on sign)
    * 0         (inactive — FC stays near zero)

Likelihood ratio turns each observation into a posterior probability
of the active state. The joint cross-feeding probability is then the
intersection of:
    "A produces X"  AND  "B consumes X"  AND  "B does NOT produce X alone"
"""

import numpy as np
from scipy.stats import norm


# ── Tunable hyperparameters ───────────────────────────────────────────────
# These are the PRIMARY sensitivity levers for the PGM.
#
# MU_PROD / MU_CONS  — expected log2FC signal for an active state.
#   Matched to FC_THRESH = 1.0 in constants.py (= 2-fold change).
#   Raise MU_PROD / lower MU_CONS to require stronger signals before
#   calling a metabolite produced/consumed.
#
# MIN_SIGMA  — floor on the likelihood width (prevents delta-function
#   posteriors for perfectly reproducible 2-rep samples).
#   Lower = stricter; 1e-3 lets low-CV samples dominate as expected.
#
# To tune: adjust these values and rerun the pipeline. The PGM case
# labels (cf_case) and cf_probability will change; the pattern-based
# integer score will NOT (it uses FC_THRESH directly).
MU_PROD   = 1.0    # expected log2FC for production  (+1 = 2-fold rise)
MU_CONS   = -1.0   # expected log2FC for consumption (-1 = 2-fold drop)
MIN_SIGMA = 1e-3   # likelihood width floor


def prob_latent(fc, sigma, mu_active):
    """
    P(state = active | observed fc)  via Gaussian likelihood ratio:

            f(fc; mu_active, sigma)
    P =  ---------------------------------------
         f(fc; mu_active, sigma) + f(fc; 0, sigma)

    A value of 0.5 means no information (NaN fc, zero sigma, or exactly
    equidistant from both means).
    """
    if fc is None or np.isnan(fc) or sigma <= 0:
        return 0.5

    p_active   = norm.pdf(fc, loc=mu_active, scale=sigma)
    p_inactive = norm.pdf(fc, loc=0.0,       scale=sigma)
    denom      = p_active + p_inactive
    if denom == 0 or not np.isfinite(denom):
        return 0.5
    return float(p_active / denom)


def infer_states(fc_s, fc_p, fc_m, cv_s, cv_p, cv_m,
                 mu_prod=MU_PROD, mu_cons=MU_CONS):
    """
    Compute the three posterior probabilities used downstream.

    Parameters
    ----------
    fc_s, cv_s : log2FC and CV for the SOLO step (A alone vs blank)
    fc_p, cv_p : log2FC and CV for the PARTNER step (B arrives, vs A-solo)
    fc_m, cv_m : log2FC and CV for the MIRROR SOLO (B alone vs blank)
    mu_prod    : float, optional
        Expected log2FC for a produced metabolite. Default: MU_PROD (1.0).
        Override to tune PGM sensitivity without editing the module.
    mu_cons    : float, optional
        Expected log2FC for a consumed metabolite. Default: MU_CONS (-1.0).

    Returns
    -------
    p_prod_A, p_cons_B, p_prod_B ∈ [0, 1]
    """
    sigma_s = max(float(cv_s) if not np.isnan(cv_s) else MIN_SIGMA, MIN_SIGMA)
    sigma_p = max(float(cv_p) if not np.isnan(cv_p) else MIN_SIGMA, MIN_SIGMA)
    sigma_m = max(float(cv_m) if not np.isnan(cv_m) else MIN_SIGMA, MIN_SIGMA)

    p_prod_A = prob_latent(fc_s, sigma_s, mu_prod)
    p_cons_B = prob_latent(fc_p, sigma_p, mu_cons)
    p_prod_B = prob_latent(fc_m, sigma_m, mu_prod)

    return p_prod_A, p_cons_B, p_prod_B


def compute_cf_probability(p_prod_A, p_cons_B, p_prod_B):
    """
    P(cross-feeding) = P(A produces) \u2227 P(B consumes) \u2227 \u00ACP(B produces alone)

    Assumes conditional independence of the three latent events given
    the metabolite — acceptable simplification given the sparse (2
    technical replicate) experimental design.
    """
    return float(p_prod_A) * float(p_cons_B) * (1.0 - float(p_prod_B))
