import pymc as pm
import pytensor.tensor as at
import numpy as np

def build_subject_model(n_stim, y_obs):

    N = len(y_obs)

    # Model
    with pm.Model() as model:

        # --- Bias & scaling parameters ---
        alpha = pm.Normal("alpha", mu=1.0, sigma=0.5)      # multiplicative bias
        beta  = pm.Normal("beta",  mu=1.0, sigma=0.3)      # compression/expansion

        # --- Noise parameters ---
        sigma_0 = pm.HalfNormal("sigma_0", sigma=1.0)      # baseline noise
        sigma_1 = pm.HalfNormal("sigma_1", sigma=0.3)      # Weber-like scaling

        # --- Latent perceived numerosity mean ---
        mu_hat = alpha * at.power(n_stim, beta)

        # --- Magnitude-dependent noise ---
        sigma_hat = sigma_0 + sigma_1 * n_stim

        pm.Deterministic("mu_hat", mu_hat)
        pm.Deterministic("sigma_hat", sigma_hat)

        # Discretized likelihood
        # Probability that latent value falls into bin
        # [y - 0.5, y + 0.5]
        # Discretized likelihood
        dist = pm.Normal.dist(mu_hat, sigma_hat)

        log_cdf_upper = pm.logcdf(dist, y_obs + 0.5)
        log_cdf_lower = pm.logcdf(dist, y_obs - 0.5)

        log_p = log_cdf_upper + at.log1p(
            -pm.math.exp(log_cdf_lower - log_cdf_upper)
        )

        pm.Potential("likelihood", at.sum(log_p))

        # Generate posterior predictions
        y_latent = pm.Normal(
            "y_latent",
            mu=mu_hat,
            sigma=sigma_hat,
            shape=N
        )

        pm.Deterministic(
            "y_pred",
            at.round(y_latent)
        )

    return model

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

def build_hierarchical_prior(
    n_stim,
    y_obs,
    subject_id,
    s_param_ph,  # subject-level: group id (0=NH, 1=PH)
    entity,      # trial-level: 0=object, 1=human
    dose=None    # optional; not used here
):
    # Dimensions
    n_trials = len(y_obs)
    n_subjects = len(np.unique(subject_id))
    n_groups = len(np.unique(s_param_ph))  # should be 2

    # Build coords
    coords = {
        "trial": np.arange(n_trials),
        "subject": np.arange(n_subjects),
        "group": np.array(["NH", "PH"]),
        "entity_type": ["object", "human"]
    }

    # Map subjects to groups
    subject_to_group = s_param_ph  # shape: (n_subjects,)
    group_idx_per_trial = subject_to_group[subject_id]  # shape: (n_trials,)
    entity = entity.astype(int)

    with pm.Model(coords=coords) as model:
        # ---------- Group-level priors for the prior exponent (alpha_prior) ----------
        # Prior exponent controls internal belief: P(k) ∝ 1/k^alpha_prior
        # Typical value: ~2.0 (from Cheyette & Piantadosi)
        log_alpha_prior_group_mu = pm.Normal(
            "log_alpha_prior_group_mu", mu=np.log(2.0), sigma=0.3, dims="group"
        )
        log_alpha_prior_group_sigma = pm.HalfNormal(
            "log_alpha_prior_group_sigma", sigma=0.2, dims="group"
        )

        # ---------- Subject-level prior exponents ----------
        log_alpha_prior_subj = pm.Normal(
            "log_alpha_prior_subj",
            mu=log_alpha_prior_group_mu[subject_to_group],
            sigma=log_alpha_prior_group_sigma[subject_to_group],
            dims="subject"
        )
        alpha_prior_subj = pm.Deterministic("alpha_prior_subj", at.exp(log_alpha_prior_subj))

        # ---------- Entity-Group modulation of the prior exponent ----------
        delta_log_alpha_prior = pm.Normal(
            "delta_log_alpha_prior",
            mu=0.0,
            sigma=0.3,
            dims=("group", "entity_type")
        )
        log_alpha_prior_trial = (
            log_alpha_prior_subj[subject_id] 
            + delta_log_alpha_prior[group_idx_per_trial, entity]
        )
        alpha_prior_trial = pm.Deterministic("alpha_prior_trial", at.exp(log_alpha_prior_trial))

        alpha_trial = pm.Deterministic("alpha_scale", 2.0 / alpha_prior_trial)

        # ---------- Beta (compression/expansion) remains as in original model ----------
        beta_group_mu = pm.Normal("beta_group_mu", mu=1.0, sigma=0.2, dims="group")
        beta_group_sigma = pm.HalfNormal("beta_group_sigma", sigma=0.1, dims="group")

        beta_subj = pm.Normal(
            "beta_subj",
            mu=beta_group_mu[subject_to_group],
            sigma=beta_group_sigma[subject_to_group],
            dims="subject"
        )

        delta_beta_group_entity = pm.Normal(
            "delta_beta_group_entity", 
            mu=0.0, 
            sigma=0.1, 
            dims=("group", "entity_type")
        )
        delta_beta_trial = delta_beta_group_entity[group_idx_per_trial, entity]
        beta_trial = beta_subj[subject_id] + delta_beta_trial

        # ---------- Noise parameters ----------
        sigma_0_group = pm.HalfNormal("sigma_0_group", sigma=1.0, dims="group")
        sigma_1_group = pm.HalfNormal("sigma_1_group", sigma=0.3, dims="group")

        sigma_0_trial = sigma_0_group[group_idx_per_trial]
        sigma_1_trial = sigma_1_group[group_idx_per_trial]

        # ---------- Latent mean and std ----------
        mu_hat = alpha_trial * at.power(n_stim, beta_trial)
        sigma_hat = sigma_0_trial + sigma_1_trial * n_stim

        pm.Deterministic("mu_hat", mu_hat, dims="trial")
        pm.Deterministic("sigma_hat", sigma_hat, dims="trial")

        # ---------- Prior predictive ----------
        y_pp = pm.Normal("y_pp", mu=mu_hat, sigma=sigma_hat, dims="trial")
        y_pred_pp = pm.Deterministic("y_pred_pp", at.round(y_pp), dims="trial")

        # ---------- Discretized likelihood ----------
        dist = pm.Normal.dist(mu=mu_hat, sigma=sigma_hat)
        log_cdf_upper = pm.logcdf(dist, y_obs + 0.5)
        log_cdf_lower = pm.logcdf(dist, y_obs - 0.5)
        log_p = log_cdf_upper + at.log1p(-at.exp(log_cdf_lower - log_cdf_upper))
        pm.Potential("likelihood", at.sum(log_p))

        # ---------- Posterior predictive ----------
        y_latent = pm.Normal("y_latent", mu=mu_hat, sigma=sigma_hat, dims="trial")
        pm.Deterministic("y_pred", at.round(y_latent), dims="trial")

    return model