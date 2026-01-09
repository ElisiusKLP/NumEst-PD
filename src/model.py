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

    # map subjects to groups for the hierarchy
    subject_group_idx = np.array(s_param_ph, dtype='int32')     
    # map data inputs to int32 for indexing
    subject_id_idx = np.array(subject_id, dtype='int32')
    entity_idx = np.array(entity, dtype='int32')

    # Build coords
    coords = {
        "trial": np.arange(n_trials),
        "subject": np.arange(n_subjects),
        "group": np.array(["NH", "PH"]),  # or use 0,1,2 if preferred
        "entity_type": ["object", "human"]
    }
    
    with pm.Model(coords=coords) as model:
        # --- Data Containers ---
        # Allows changing data for predictions later without rebuilding graph
        subject_id_pt = pm.Data("subject_id_data", subject_id_idx, dims="trial")
        entity_pt = pm.Data("entity_data", entity_idx, dims="trial")
        n_stim_pt = pm.Data("n_stim_data", n_stim, dims="trial")

        # --- 1. Alpha (Bias/Salience) ---
        # Structure: Group x Entity -> Subject x Entity
        # We assume different groups react to entities differently (Hypothesis check)
        log_alpha_group = pm.Normal("log_alpha_group", mu=0.0, sigma=0.1, dims=("group", "entity_type"))
        log_alpha_sigma = pm.HalfNormal("log_alpha_sigma", sigma=0.1, dims=("group", "entity_type"))

        # Subject level: dims=(subject, entity_type)
        # Subjects are centered on THEIR group's mean for that entity
        log_alpha_subj = pm.Normal(
            "log_alpha_subj", 
            mu=log_alpha_group[subject_group_idx, :], # Broadcasting group mean to subjects
            sigma=log_alpha_sigma[subject_group_idx, :],
            dims=("subject", "entity_type")
        )
        
        # Select specific alpha for each trial
        log_alpha_trial = log_alpha_subj[subject_id_pt, entity_pt]

        # --- 2. Beta (Compression/Priors) ---
        # Structure: Group x Entity -> Subject x Entity
        # THIS is where your "Strong Prior" hypothesis lives.
        
        beta_group = pm.Normal("beta_group", mu=1, sigma=0.1, dims=("group", "entity_type"))
        beta_sigma = pm.HalfNormal("beta_sigma", sigma=0.05, dims=("group", "entity_type"))
        
        beta_subj = pm.Normal(
            "beta_subj",
            mu=beta_group[subject_group_idx, :],
            sigma=beta_sigma[subject_group_idx, :],
            dims=("subject", "entity_type")
        )
        beta_trial = beta_subj[subject_id_pt, entity_pt]


        # --- 3. Prediction (Power Law) ---
        mu_hat_log = log_alpha_trial + beta_trial * at.log(n_stim_pt)
        
        # --- 4. Noise Model (Weber's Law) ---
        # Sigma scales with magnitude (scalar variability)
        # We let noise vary by Group (PH might be noisier)
        sigma_group = pm.HalfNormal("sigma_group", sigma=0.1, dims="group")
        
        # Map group noise to trial
        # Note: We index group by subject, then subject by trial
        trial_group_idx = pm.Data("trial_group_idx", subject_group_idx[subject_id_idx], dims="trial")
        sigma_trial = sigma_group[trial_group_idx]

        # ---------- Set prior predictive node -------
        y_pp = pm.LogNormal("y_pp", mu=mu_hat_log, sigma=sigma_trial, dims="trial")
        y_pred_pp = pm.Deterministic("y_pred_pp", at.round(y_pp), dims="trial")

        # --- 5. Likelihood (Discretized Normal) ---
        # This handles the integer nature of the response (rounding)
        dist = pm.LogNormal.dist(mu=mu_hat_log, sigma=sigma_trial)

        # Safety: LogNormal is undefined for negative numbers.
        # clip the lower bound for the CDF calculation.
        lower_bound = at.maximum(y_obs - 0.5, 1e-4)
        upper_bound = y_obs + 0.5

        log_cdf_upper = pm.logcdf(dist, upper_bound)
        log_cdf_lower = pm.logcdf(dist, lower_bound)
        log_p = log_cdf_upper + at.log1p(-at.exp(log_cdf_lower - log_cdf_upper))
        
        pm.Potential("likelihood", at.sum(log_p))

        # --- 6. Generated Quantities ---
        pm.Deterministic("mu_hat_log", mu_hat_log, dims="trial")
        
        # Posterior predictive for checking model fit
        y_log_latent = pm.LogNormal("y_latent", mu=mu_hat_log, sigma=sigma_trial, dims="trial")
        pm.Deterministic("y_pred", at.round(y_log_latent), dims="trial")

    return model

def build_hierarchical_lnlik(
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

    # map subjects to groups for the hierarchy
    subject_group_idx = np.array(s_param_ph, dtype='int32')     
    # map data inputs to int32 for indexing
    subject_id_idx = np.array(subject_id, dtype='int32')
    entity_idx = np.array(entity, dtype='int32')

    # Build coords
    coords = {
        "trial": np.arange(n_trials),
        "subject": np.arange(n_subjects),
        "group": np.array(["NH", "PH"]),  # or use 0,1,2 if preferred
        "entity_type": ["object", "human"]
    }
    
    with pm.Model(coords=coords) as model:
        # --- Data Containers ---
        # Allows changing data for predictions later without rebuilding graph
        subject_id_pt = pm.Data("subject_id_data", subject_id_idx, dims="trial")
        entity_pt = pm.Data("entity_data", entity_idx, dims="trial")
        n_stim_pt = pm.Data("n_stim_data", n_stim, dims="trial")

        # --- 1. Alpha (Bias/Salience) ---
        # Structure: Group x Entity -> Subject x Entity
        # We assume different groups react to entities differently (Hypothesis check)
        log_alpha_group = pm.Normal("log_alpha_mu", mu=0.8, sigma=0.7, dims=("group", "entity_type"))
        log_alpha_sigma = pm.HalfNormal("log_alpha_sigma", sigma=0.3, dims=("group", "entity_type"))

        # Subject level: dims=(subject, entity_type)
        # Subjects are centered on THEIR group's mean for that entity
        log_alpha_subj = pm.Normal(
            "log_alpha_subj", 
            mu=log_alpha_group[subject_group_idx, :], # Broadcasting group mean to subjects
            sigma=log_alpha_sigma[subject_group_idx, :],
            dims=("subject", "entity_type")
        )
        
        # Select specific alpha for each trial
        log_alpha_trial = log_alpha_subj[subject_id_pt, entity_pt]

        # --- 2. Beta (Compression/Priors) ---
        # Structure: Group x Entity -> Subject x Entity
        
        beta_group = pm.Normal("beta_mu", mu=.98, sigma=0.1, dims=("group", "entity_type"))
        beta_sigma = pm.HalfNormal("beta_sigma", sigma=0.1, dims=("group", "entity_type"))
        

        beta_subj = pm.Normal(
            "beta_subj",
            mu=beta_group[subject_group_idx, :],
            sigma=beta_sigma[subject_group_idx, :],
            dims=("subject", "entity_type")
        )
        beta_trial = beta_subj[subject_id_pt, entity_pt]


        # --- 3. Prediction (Power Law) ---
        mu_hat_log = log_alpha_trial + beta_trial * at.log(n_stim_pt)
        
        # --- 4. Noise Model (Weber's Law) ---
        # Sigma scales with magnitude (scalar variability)
        # We let noise vary by Group (PH might be noisier)
        sigma_group = pm.HalfNormal("sigma_group", sigma=0.1, dims="group")
        
        # Map group noise to trial
        # Note: We index group by subject, then subject by trial
        trial_group_idx = pm.Data("trial_group_idx", subject_group_idx[subject_id_idx], dims="trial")
        sigma_trial = sigma_group[trial_group_idx]

        # --- 5. Likelihood (Discretized Normal) ---
        # This handles the integer nature of the response (rounding)
        y = pm.LogNormal(
            "y",
            mu=mu_hat_log,
            sigma=sigma_trial,
            observed=y_obs,
            dims="trial"
        )

        # --- 6. Generated Quantities ---
        pm.Deterministic("mu_hat_log", mu_hat_log, dims="trial")
        
        pm.Deterministic("y_pred", at.round(y), dims="trial")

    return model

def build_hierarchical_with_prior(
    n_stim,
    y_obs,
    subject_id,
    s_param_ph,
    entity,
    dose=None
):
    n_trials = len(y_obs)
    n_subjects = len(np.unique(subject_id))
    
    # Ensure strict integer types
    subject_to_group_idx = np.array(s_param_ph, dtype='int32')
    entity_idx = np.array(entity, dtype='int32')
    subject_id_idx = np.array(subject_id, dtype='int32')

    coords = {
        "trial": np.arange(n_trials),
        "subject": np.arange(n_subjects),
        "group": ["NH", "PH"],
        "entity_type": ["object", "human"]
    }

    with pm.Model(coords=coords) as model:
        # Data containers for graph stability
        subject_id_pt = pm.Data("subject_id_data", subject_id_idx, dims="trial")
        entity_pt = pm.Data("entity_data", entity_idx, dims="trial")
        n_stim_pt = pm.Data("n_stim_data", n_stim, dims="trial")
        
        # 1. Internal Noise (w)
        w_group = pm.HalfNormal("w_group", sigma=0.1, dims="group")
        w_subj = pm.HalfNormal("w_subj", sigma=w_group[subject_to_group_idx], dims="subject")
        w_trial = w_subj[subject_id_pt]

        # 2. Prior Parameter (Gamma)
        gamma_group_entity = pm.Normal("gamma_group_entity", mu=1.0, sigma=0.5, dims=("group", "entity_type"))
        gamma_subj_entity = pm.Normal("gamma_subj_entity", mu=gamma_group_entity[subject_to_group_idx], sigma=0.3, dims=("subject", "entity_type"))
        gamma_trial = gamma_subj_entity[subject_id_pt, entity_pt]

        # 3. Derive Beta
        denom_raw = 1.0 + (w_trial**2) * gamma_trial
        denom_safe = at.maximum(denom_raw, 0.05) 
        beta_trial = 1.0 / denom_safe

        # 4. Bias (Alpha)
        log_alpha_group_entity = pm.Normal("log_alpha_group", mu=0.0, sigma=0.3, dims=["group", "entity_type"])
        log_alpha_subj = pm.Normal("log_alpha_subj", mu=log_alpha_group_entity[subject_to_group_idx], sigma=0.2, dims=("subject", "entity_type"))
        alpha_trial = at.exp(log_alpha_subj[subject_id_pt, entity_pt])

        # 5. Prediction
        mu_hat = alpha_trial * at.power(n_stim_pt, beta_trial)

        # 6. Observation Noise
        phi = pm.HalfNormal("phi", sigma=0.15, dims="group")
        total_weber_trial = at.sqrt(w_trial**2 + phi[subject_to_group_idx[subject_id_idx]]**2)
        
        # FIX: Ensure sigma_hat is never exactly zero
        sigma_hat = at.maximum(total_weber_trial * mu_hat, 1e-4)

        # ---------- Set prior predictive node -------
        y_pp = pm.Normal("y_pp", mu=mu_hat, sigma=sigma_hat, dims="trial")
        y_pred_pp = pm.Deterministic("y_pred_pp", at.round(y_pp), dims="trial")

        # 7. Likelihood using CustomDist
        # This allows observed=y_obs AND enables sample_posterior_predictive automatically
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

        # Deterministics
        pm.Deterministic("mu_hat", mu_hat, dims="trial")
        pm.Deterministic("sigma_hat", sigma_hat, dims="trial")
        pm.Deterministic("beta_trial", beta_trial, dims="trial")

    return model