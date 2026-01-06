import pymc as pm
import pytensor.tensor as at
import numpy as np

def build_hierarchical(
    n_stim,
    y_obs,
    subject_id,
    s_param_ph,  # subject-level: group id (0,1)
    entity,      # trial-level: 0=object, 1=human
    dose=None    # optional; not used here but kept for signature
):
    # Dimensions
    n_trials = len(y_obs)
    n_subjects = len(np.unique(subject_id))
    n_groups = len(np.unique(s_param_ph))  # should be 2

    # Build coords
    coords = {
        "trial": np.arange(n_trials),
        "subject": np.arange(n_subjects),
        "group": np.array(["NH", "PH"]),  # or use 0,1,2 if preferred
        "entity_type": ["object", "human"]
    }

    # Map subjects to groups (s_param_ph is per subject)
    # Ensure s_param_ph is aligned with subject index
    subject_to_group = s_param_ph  # shape: (n_subjects,)
    group_idx_per_trial = subject_to_group[subject_id]  # shape: (n_trials,)

    entity = entity.astype(int)  # ensure integer

    with pm.Model(coords=coords) as model:
        # ---------- Group-level priors ----------
        # Group-level hyperpriors for alpha (bias) and beta (compression)
        alpha_group_mu = pm.Normal("alpha_group_mu", mu=1.0, sigma=0.3, dims="group")
        alpha_group_sigma = pm.HalfNormal("alpha_group_sigma", sigma=0.2, dims="group")

        beta_group_mu = pm.Normal("beta_group_mu", mu=1.0, sigma=0.2, dims="group")
        beta_group_sigma = pm.HalfNormal("beta_group_sigma", sigma=0.1, dims="group")

        # ---------- Subject-level parameters ----------
        # Each subject draws from their group's distribution
        alpha_subj = pm.Normal(
            "alpha_subj",
            mu=alpha_group_mu[subject_to_group],
            sigma=alpha_group_sigma[subject_to_group],
            dims="subject"
        )
        beta_subj = pm.Normal(
            "beta_subj",
            mu=beta_group_mu[subject_to_group],
            sigma=beta_group_sigma[subject_to_group],
            dims="subject"
        )

        # ---------- Entity-Group interaction modulation (human vs object) ----------
        # We model additional bias when estimating humans vs objects X NH vs PH
        delta_alpha_group_entity = pm.Normal(
            "delta_alpha_group_entity",
            mu=0.0,
            sigma=0.2,
            dims=("group", "entity_type")
        )
        delta_beta_group_entity = pm.Normal(
            "delta_beta_group_entity", 
            mu=0.0, 
            sigma=0.1, 
            dims=("group", "entity_type")
        )

        # Index per trial: (n_trials,) → pulls (group, entity) pair
        delta_alpha_trial = delta_alpha_group_entity[group_idx_per_trial, entity]
        delta_beta_trial = delta_beta_group_entity[group_idx_per_trial, entity]

        # Combine subject + interaction effect
        alpha_trial = alpha_subj[subject_id] + delta_alpha_trial
        beta_trial = beta_subj[subject_id] + delta_beta_trial

        # ---------- Noise parameters (could also be hierarchical, but starting simple) ----------
        sigma_0_group = pm.HalfNormal("sigma_0_group", sigma=1.0, dims="group")
        sigma_1_group = pm.HalfNormal("sigma_1_group", sigma=0.3, dims="group")

        sigma_0_trial = sigma_0_group[group_idx_per_trial]
        sigma_1_trial = sigma_1_group[group_idx_per_trial]

        # ---------- Latent mean and std ----------
        mu_hat = alpha_trial * at.power(n_stim, beta_trial)
        sigma_hat = sigma_0_trial + sigma_1_trial * n_stim

        pm.Deterministic("mu_hat", mu_hat, dims="trial")
        pm.Deterministic("sigma_hat", sigma_hat, dims="trial")

        # ---------- Set prior predictive node -------
        y_pp = pm.Normal("y_pp", mu=mu_hat, sigma=sigma_hat, dims="trial")
        y_pred_pp = pm.Deterministic("y_pred_pp", at.round(y_pp), dims="trial")

        # ---------- Discretized likelihood ----------
        dist = pm.Normal.dist(mu=mu_hat, sigma=sigma_hat)
        log_cdf_upper = pm.logcdf(dist, y_obs + 0.5)
        log_cdf_lower = pm.logcdf(dist, y_obs - 0.5)
        # Stable computation of log(P(y - 0.5 < X < y + 0.5))
        log_p = log_cdf_upper + at.log1p(-at.exp(log_cdf_lower - log_cdf_upper))
        pm.Potential("likelihood", at.sum(log_p))

        # ---------- Posterior predictive ----------
        y_latent = pm.Normal("y_latent", mu=mu_hat, sigma=sigma_hat, dims="trial")
        pm.Deterministic("y_pred", at.round(y_latent), dims="trial")

    return model