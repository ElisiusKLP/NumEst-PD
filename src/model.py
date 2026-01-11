import pymc as pm
import pytensor.tensor as at
import numpy as np

def build_null_lnlik(
    n_stim,
    y_obs,
    subject_id,
    s_param_ph,
    entity,
    dose=None
):
    n_trials = len(y_obs)

    coords = {
        "trial": np.arange(n_trials)
    }
    
    with pm.Model(coords=coords) as model:
        n_stim_pt = pm.Data("n_stim_data", n_stim, dims="trial")

        log_alpha = pm.Normal("log_alpha", mu=0.8, sigma=0.7)

        beta = pm.Normal("beta", mu=0.98, sigma=0.1)

        mu_hat_log = log_alpha + beta * at.log(n_stim_pt)
        
        sigma = pm.HalfNormal("sigma", sigma=0.1)

        y = pm.LogNormal(
            "y",
            mu=mu_hat_log,
            sigma=sigma,
            observed=y_obs,
            dims="trial"
        )

        pm.Deterministic("mu_hat_log", mu_hat_log, dims="trial")
        
        pm.Deterministic("y_pred", at.round(y), dims="trial")

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
