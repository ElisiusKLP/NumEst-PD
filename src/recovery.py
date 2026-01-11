
import numpy as np
import pymc as pm
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

# --------------------------------------------------------------------------
# 1. UPDATED SIMULATION FUNCTION
# --------------------------------------------------------------------------
def simulate_hierarchical_data(
    n_subjects=20,       # Total subjects
    n_stim_minmax=[5,8], # Range of stimulus intensities
    n_reps=5,            # Repetitions per condition per subject
    true_params=None     # Dictionary of true parameters
):
    """
    Generates data consistent with the new build_hierarchical_lnlik model.
    Structure:
      - Likelihood: LogNormal
      - Mean (log-space): log_alpha + beta * log(stim)
      - Params: Matrices of shape (Group, Entity)
    """
    rng = np.random.default_rng(42)

    # --- 1. Experimental Design ---
    # Groups: 0=NH, 1=PH. Assign half to each.
    s_param_ph = np.zeros(n_subjects, dtype=int)
    s_param_ph[n_subjects // 2:] = 1  
    
    # Stimulus range
    stim_levels = range(n_stim_minmax[0], n_stim_minmax[1]+1)
    
    # Entities: 0=Object, 1=Human
    entities = [0, 1]

    # Create trial structure
    trial_list = []
    for subj_i in range(n_subjects):
        group_id = s_param_ph[subj_i]
        for ent in entities:
            for stim in stim_levels:
                for _ in range(n_reps):
                    trial_list.append({
                        'subject_id': subj_i,
                        'group_id': group_id,
                        'entity': ent,
                        'n_stim': stim
                    })
    
    df = pd.DataFrame(trial_list)
    
    # Extract arrays for vectorization
    subject_idx = df['subject_id'].values
    group_idx_per_trial = df['group_id'].values
    entity_idx = df['entity'].values
    n_stim_val = df['n_stim'].values

    # --- 2. Unpack True Parameters ---
    if true_params is None:
        raise ValueError("No true parameters provided.")

    # --- 3. Generate Subject-Level Parameters ---
    # In the new model, subjects have params for BOTH entities.
    # Shape: (n_subjects, n_entities)
    
    log_alpha_subj = np.zeros((n_subjects, 2))
    beta_subj = np.zeros((n_subjects, 2))
    
    for s in range(n_subjects):
        g = s_param_ph[s] # Subject's group
        
        # Sample subject parameters for both entities (0 and 1) based on their group priors
        for e in [0, 1]:
            # log_alpha
            mu_a = true_params['log_alpha_mu'][g, e]
            sigma_a = true_params['log_alpha_sigma'][g, e]
            log_alpha_subj[s, e] = rng.normal(loc=mu_a, scale=sigma_a)
            
            # beta
            mu_b = true_params['beta_mu'][g, e]
            sigma_b = true_params['beta_sigma'][g, e]
            beta_subj[s, e] = rng.normal(loc=mu_b, scale=sigma_b)

    # --- 4. Calculate Trial-Level Latents ---
    
    # Lookup specific params for each trial: [subject_id, entity_id]
    # This vectorizes the selection of the specific alpha/beta used in that specific trial
    trial_log_alpha = log_alpha_subj[subject_idx, entity_idx]
    trial_beta      = beta_subj[subject_idx, entity_idx]
    
    # Prediction (Power Law in Log Space)
    # Model: mu_hat_log = log_alpha + beta * log(stim)
    # Note: Ensure n_stim > 0 to avoid log(0) error
    mu_hat_log = trial_log_alpha + trial_beta * np.log(n_stim_val)
    
    # Noise Model
    # The new model has sigma specific to Group (NH vs PH)
    # trial_group_idx matches the subject group
    trial_sigma = true_params['sigma_group'][group_idx_per_trial]
    
    # --- 5. Generate Observations ---
    # Sample from LogNormal
    # PyMC/Numpy LogNormal takes mean and sigma of the underlying Normal distribution
    y_latent = rng.lognormal(mean=mu_hat_log, sigma=trial_sigma)
    
    # Round to nearest integer (matching the discrete nature of y_obs in typical psychophysics)
    y_obs = np.round(y_latent)

    # Pack for return
    data_dict = {
        'n_stim': n_stim_val,
        'y_obs': y_obs,
        'subject_id': subject_idx,
        's_param_ph': s_param_ph, # (n_subjects,)
        'entity': entity_idx,
        'dose': None
    }
    
    return data_dict, true_params

def run_sensitivity_analysis(model_builder, base_params, sensitivity_config, sim_kwargs):
    """
    Runs a One-Factor-At-A-Time (OAT) sweep.
    Adapted for the matrix-style parameters (Group x Entity).
    """
    results = []
    
    # Updated mappings to match new model dims: ("group", "entity_type")
    coord_groups = ["NH", "PH"]
    coord_entities = ["object", "human"]

    print(f"Starting Sensitivity Analysis...")
    print(f"Base Parameters keys: {list(base_params.keys())}")
    
    # Outer Loop: Which parameter are we stressing?
    for param_name, value_list in sensitivity_config.items():
        print(f"\n>>> TESTING PARAMETER: {param_name}")
        
        for step_i, test_value in enumerate(value_list):
            print(f"   Step {step_i+1}/{len(value_list)}")
            
            # 1. Construct Parameters
            current_params = base_params.copy()
            current_params[param_name] = test_value
            
            # 2. Simulate
            sim_data, _ = simulate_hierarchical_data(
                n_subjects=sim_kwargs.get("n_subjects", 40),
                n_stim_minmax=sim_kwargs.get("n_stim_minmax", [5,8]),
                n_reps=sim_kwargs.get("n_reps", 5),
                true_params=current_params
            )
            
            # 3. Fit
            try:
                # Build model using unpacked dictionary
                model = model_builder(**sim_data)
                
                with model:
                    # Reduced settings for speed in sensitivity checks (adjust as needed)
                    idata = pm.sample(
                        sim_kwargs.get("samples", 500), 
                        tune=sim_kwargs.get("tune", 500), 
                        chains=2, 
                        progressbar=True # Cleaner output
                    )
                
                n_div = int(idata.sample_stats.diverging.sum())
                summary = az.summary(idata, hdi_prob=0.95)
                
                # 4. Record Results
                all_keys_to_check = list(base_params.keys())
                
                for p_check in all_keys_to_check:
                    true_val = current_params[p_check]
                    
                    if np.isscalar(true_val):
                        # Scalar Logic
                        if p_check in summary.index:
                            results.append({
                                'varied_param': param_name,
                                'step_index': step_i,
                                'param_measured': p_check,
                                'index_name': None,
                                'true_value': true_val,
                                'est_mean': summary.loc[p_check, 'mean'],
                                'hdi_lo': summary.loc[p_check, 'hdi_2.5%'],
                                'hdi_hi': summary.loc[p_check, 'hdi_97.5%'],
                                'divergences': n_div
                            })
                    else:
                        # Array/Matrix Logic
                        true_val_arr = np.array(true_val)
                        it = np.nditer(true_val_arr, flags=['multi_index'])
                        for x in it:
                            idx_tuple = it.multi_index # e.g. (0, 1)
                            
                            # Construct PyMC parameter name candidates
                            # 1. Raw index: "log_alpha_mu[0,1]"
                            # 2. Label index: "log_alpha_mu[NH, human]"
                            
                            candidates = []
                            idx_str_raw = ",".join(map(str, idx_tuple))
                            candidates.append(f"{p_check}[{idx_str_raw}]")
                            
                            # Map (0,1) -> (NH, human)
                            if len(idx_tuple) == 2:
                                g_i, e_i = idx_tuple
                                if g_i < len(coord_groups) and e_i < len(coord_entities):
                                    g_lbl = coord_groups[g_i]
                                    e_lbl = coord_entities[e_i]
                                    candidates.append(f"{p_check}[{g_lbl}, {e_lbl}]")
                                    candidates.append(f"{p_check}[{g_lbl},{e_lbl}]") # sometimes spaces differ
                            
                            elif len(idx_tuple) == 1:
                                # For sigma_group[NH]
                                if idx_tuple[0] < len(coord_groups):
                                    candidates.append(f"{p_check}[{coord_groups[idx_tuple[0]]}]")

                            # Find match in summary
                            for cand in candidates:
                                if cand in summary.index:
                                    results.append({
                                        'varied_param': param_name,
                                        'step_index': step_i,
                                        'param_measured': p_check,
                                        'index_name': cand,
                                        'true_value': float(x),
                                        'est_mean': summary.loc[cand, 'mean'],
                                        'hdi_lo': summary.loc[cand, 'hdi_2.5%'],
                                        'hdi_hi': summary.loc[cand, 'hdi_97.5%'],
                                        'divergences': n_div
                                    })
                                    break
            except Exception as e:
                print(f"!!! MODEL FAILURE at {param_name} Step {step_i}: {e}")
                
    return pd.DataFrame(results)

def plot_sensitivity(df_results):
    """
    Creates one figure per Varied Parameter.
    """
    # Get list of unique parameters we stressed
    varied_params = df_results['varied_param'].unique()
    
    for v_param in varied_params:
        # Filter: Only look at rows where THIS parameter was the one being varied
        # AND we only care about the recovery of THIS parameter (ignoring cross-effects for the plot)
        subset = df_results[
            (df_results['varied_param'] == v_param) & 
            (df_results['param_measured'] == v_param)
        ].copy()
        
        if subset.empty:
            continue
            
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # We might have multiple indices (e.g., alpha[0], alpha[1]). 
        # Let's plot them with different markers/colors.
        indices = subset['index_name'].unique()
        
        # If index_name is None (scalars), handle gracefully
        if len(indices) == 0 or (len(indices) == 1 and indices[0] is None):
            indices = ['Scalar']
            subset['index_name'] = 'Scalar'

        for idx_name in indices:
            sub_idx = subset[subset['index_name'] == idx_name]
            
            # Error bars
            yerr = [
                sub_idx['est_mean'] - sub_idx['hdi_lo'],
                sub_idx['hdi_hi'] - sub_idx['est_mean']
            ]
            
            ax.errorbar(
                sub_idx['true_value'], 
                sub_idx['est_mean'], 
                yerr=yerr, 
                fmt='-o', 
                label=idx_name,
                capsize=5,
                alpha=0.8
            )
            
        # Identity line
        all_vals = np.concatenate([subset['true_value'], subset['est_mean']])
        min_v, max_v = all_vals.min(), all_vals.max()
        pad = (max_v - min_v) * 0.1
        ax.plot([min_v-pad, max_v+pad], [min_v-pad, max_v+pad], 'k--', alpha=0.3, label="Ideal")
        
        ax.set_title(f"Sensitivity Analysis: Varying {v_param}")
        ax.set_xlabel("True Value")
        ax.set_ylabel("Estimated Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add divergence count as text if high
        total_div = subset['divergences'].max()
        if total_div > 0:
            plt.figtext(0.15, 0.85, f"Max Divergences: {total_div}", color='red', weight='bold')

        plt.tight_layout()
        plt.show()