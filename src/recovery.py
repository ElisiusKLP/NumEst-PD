
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import math

def simulate_hierarchical_data(
    n_subjects=20,       # Total subjects
    n_stim_minmax=[5,8],     # How many stimulus intensities
    n_reps=5,            # Repetitions per condition per subject
    true_params=None     # Dictionary of true parameters
):
    """
    Generates data consistent with the provided build_hierarchical model.
    """
    rng = np.random.default_rng(42)

    # --- 1. Experimental Design ---
    # Groups: 0=NH, 1=PH. Assign half to each.
    s_param_ph = np.zeros(n_subjects, dtype=int)
    s_param_ph[n_subjects // 2:] = 1  
    
    # Stimulus range (e.g., 1 to 10 or 10 to 100)
    stim_levels = range(n_stim_minmax[0], n_stim_minmax[1]+1)
    
    # Entities: 0=Object, 1=Human
    entities = [0, 1]

    # Create trial structure
    # We need to build long-format arrays for subject_id, n_stim, entity, etc.
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
    n_trials = len(df)
    
    # Extract arrays for vectorization
    subject_idx = df['subject_id'].values
    group_idx_per_trial = df['group_id'].values
    entity_idx = df['entity'].values
    n_stim_val = df['n_stim'].values

    # --- 2. Unpack True Parameters ---
    # Default parameters if none provided
    if true_params is None:
        raise ValueError("No true parameters provided.")

    # --- 3. Generate Subject-Level Parameters ---
    # Sample alpha_subj and beta_subj from group distributions
    alpha_subj = np.zeros(n_subjects)
    beta_subj = np.zeros(n_subjects)
    
    for s in range(n_subjects):
        g = s_param_ph[s]
        alpha_subj[s] = rng.normal(
            loc=true_params['alpha_group_mu'][g], 
            scale=true_params['alpha_group_sigma'][g]
        )
        beta_subj[s] = rng.normal(
            loc=true_params['beta_group_mu'][g], 
            scale=true_params['beta_group_sigma'][g]
        )

    # --- 4. Calculate Trial-Level Latents ---
    # Get subject baseline for every trial
    trial_alpha_subj = alpha_subj[subject_idx]
    trial_beta_subj  = beta_subj[subject_idx]
    
    # Get interaction deltas for every trial
    # Indexing: [group_idx_per_trial, entity_idx]
    trial_delta_alpha = true_params['delta_alpha_group_entity'][group_idx_per_trial, entity_idx]
    trial_delta_beta  = true_params['delta_beta_group_entity'][group_idx_per_trial, entity_idx]
    
    # Combine
    alpha_trial = trial_alpha_subj + trial_delta_alpha
    beta_trial  = trial_beta_subj + trial_delta_beta
    
    # Power law mean: alpha * stim ^ beta
    mu_hat = alpha_trial * np.power(n_stim_val, beta_trial)
    
    # Linear noise: sigma_0 + sigma_1 * stim
    sigma_hat = true_params['sigma_0_group'] + true_params['sigma_1_group'] * n_stim_val
    
    # --- 5. Generate Observations ---
    # Sample from continuous Normal
    y_latent = rng.normal(loc=mu_hat, scale=sigma_hat)
    
    # Round to nearest integer (as per the discretized likelihood in the model)
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

def run_parameter_sweep(model_builder, base_params, sweep_config, n_steps=4):
    results = []
    
    # Define mappings based on your build_hierarchical model coords
    coord_groups = ["NH", "PH"]
    coord_entities = ["object", "human"]

    print(f"Starting Parameter Sweep over {n_steps} steps...")

    for i in range(n_steps):
        print(f"\n--- Step {i+1}/{n_steps} ---")
        
        # 1. Update Params for this step
        current_params = base_params.copy()
        for key, val_list in sweep_config.items():
            current_params[key] = val_list[i]
            
        # 2. Simulate
        sim_data, truths = simulate_hierarchical_data(
            n_subjects=118, n_stim_minmax=[5,8], n_reps=10, 
            true_params=current_params
        )
        
        # 3. Fit
        model = model_builder(**sim_data)
        with model:
            # Low samples for speed; increase for precision
            idata = pm.sample(1000, tune=1000, chains=2, progressbar=True)
            
        # 4. Extract Summary
        summary = az.summary(idata, hdi_prob=0.95)
        
        # 5. Compare True vs Estimated
        for param_name in sweep_config.keys():
            true_val = current_params[param_name]
            
            # --- SCALAR PARAMETERS (e.g., sigma_0) ---
            if np.isscalar(true_val):
                if param_name in summary.index:
                    est_mean = summary.loc[param_name, 'mean']
                    hdi_lo = summary.loc[param_name, 'hdi_2.5%']
                    hdi_hi = summary.loc[param_name, 'hdi_97.5%']
                    
                    results.append({
                        'step': i, 'param': param_name, 'index': None,
                        'true_value': true_val, 'est_mean': est_mean,
                        'hdi_lo': hdi_lo, 'hdi_hi': hdi_hi
                    })
                else:
                    print(f"Warning: {param_name} not found in summary.")

            # --- VECTOR/MATRIX PARAMETERS ---
            else:
                true_val = np.array(true_val)
                it = np.nditer(true_val, flags=['multi_index'])
                
                for x in it:
                    idx_tuple = it.multi_index # e.g., (0,) or (0,1)
                    idx_str_raw = ",".join(map(str, idx_tuple))
                    
                    # --- Coordinate Name Resolution ---
                    # We try to guess the ArviZ name format based on dimensions
                    candidates = []
                    
                    # Candidate 1: Integer indexing (e.g., param[0])
                    candidates.append(f"{param_name}[{idx_str_raw}]")
                    
                    # Candidate 2: Named indexing (e.g., param[NH])
                    # Logic: If it's a 1D Group param
                    if len(idx_tuple) == 1 and idx_tuple[0] < len(coord_groups):
                        g_str = coord_groups[idx_tuple[0]]
                        candidates.append(f"{param_name}[{g_str}]")
                    
                    # Candidate 3: Named indexing 2D (e.g., param[NH, object])
                    # Logic: If it's 2D Group x Entity
                    if len(idx_tuple) == 2:
                        g_idx, e_idx = idx_tuple
                        if g_idx < len(coord_groups) and e_idx < len(coord_entities):
                            g_str = coord_groups[g_idx]
                            e_str = coord_entities[e_idx]
                            # ArviZ usually uses comma+space or just comma
                            candidates.append(f"{param_name}[{g_str}, {e_str}]")
                            candidates.append(f"{param_name}[{g_str},{e_str}]")

                    # Try to find one of the candidates in the summary index
                    found = False
                    for row_name in candidates:
                        if row_name in summary.index:
                            est_mean = summary.loc[row_name, 'mean']
                            hdi_lo = summary.loc[row_name, 'hdi_2.5%']
                            hdi_hi = summary.loc[row_name, 'hdi_97.5%']
                            
                            results.append({
                                'step': i, 'param': param_name, 'index': row_name,
                                'true_value': float(x), 'est_mean': est_mean,
                                'hdi_lo': hdi_lo, 'hdi_hi': hdi_hi
                            })
                            found = True
                            break
                    
                    if not found:
                        print(f"Warning: Could not find parameter row for {param_name} indices {idx_tuple}. Checked: {candidates}")

    return pd.DataFrame(results)

def plot_sweep_results(df_results):
    """
    Plots True (X-axis) vs Estimated (Y-axis).
    Points on the diagonal line indicate perfect recovery.
    Arranged as 2 plots per row.
    """
    params = df_results['param'].unique()
    n_params = len(params)

    n_cols = 2
    n_rows = math.ceil(n_params / n_cols)

    fig, axes = plt.subplots(
        n_rows, 
        n_cols, 
        figsize=(6 * n_cols, 5 * n_rows),
        squeeze=False
    )

    axes = axes.flatten()

    for ax, p in zip(axes, params):
        subset = df_results[df_results['param'] == p]

        yerr = [
            subset['est_mean'] - subset['hdi_lo'],
            subset['hdi_hi'] - subset['est_mean']
        ]

        ax.errorbar(
            subset['true_value'],
            subset['est_mean'],
            yerr=yerr,
            fmt='o',
            capsize=3,
            label='Estimate (95% HDI)'
        )

        min_val = min(subset['true_value'].min(), subset['est_mean'].min())
        max_val = max(subset['true_value'].max(), subset['est_mean'].max())
        buffer = (max_val - min_val) * 0.1
        line_range = [min_val - buffer, max_val + buffer]

        ax.plot(line_range, line_range, 'r--', alpha=0.5, label='Identity Line')

        ax.set_title(f"Recovery Sweep: {p}")
        ax.set_xlabel("True Value")
        ax.set_ylabel("Estimated Posterior Mean")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide any unused subplots
    for ax in axes[len(params):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()

def run_sensitivity_analysis(model_builder, base_params, sensitivity_config, sim_kwargs):
    """
    Runs a One-Factor-At-A-Time (OAT) sweep.
    It holds all parameters at 'base_params' values, except for the one currently being swept.
    """
    results = []
    
    # Define mappings for coordinate resolution (same as your original)
    coord_groups = ["NH", "PH"]
    coord_entities = ["object", "human"]

    print(f"Starting Sensitivity Analysis...")
    print(f"Base Parameters: {list(base_params.keys())}")
    
    # Outer Loop: Which parameter are we stressing?
    for param_name, value_list in sensitivity_config.items():
        print(f"\n>>> TESTING PARAMETER: {param_name}")
        
        # Inner Loop: The levels of stress (Low -> High)
        for step_i, test_value in enumerate(value_list):
            print(f"   Step {step_i+1}/{len(value_list)}")
            
            # 1. Construct Parameters: Start with Base, Overwrite Target
            current_params = base_params.copy()
            current_params[param_name] = test_value
            
            # 2. Simulate
            # Use sim_kwargs to keep code clean
            sim_data, _ = simulate_hierarchical_data(
                n_subjects=sim_kwargs.get("n_subjects", 50),
                n_stim_minmax=sim_kwargs.get("n_stim_minmax", [5,8]),
                n_reps=sim_kwargs.get("n_reps", 10),
                true_params=current_params
            )
            
            # 3. Fit
            try:
                model = model_builder(**sim_data)
                with model:
                    # Adjust samples/tune as needed for speed vs precision
                    idata = pm.sample(sim_kwargs.get("samples", 500), tune=sim_kwargs.get("tune", 500), chains=2, progressbar=True)
                
                # Check Divergences (Crucial for "Why did it fail?")
                n_div = int(idata.sample_stats.diverging.sum())
                
                # 4. Extract Summary
                summary = az.summary(idata, hdi_prob=0.95)
                
                # 5. Record Results (Focusing on the parameter being varied)
                # We also record ALL parameters to check for "leakage" (e.g., does changing beta break sigma?)
                # But to keep the log manageable, let's iterate through the base_params keys
                
                all_keys_to_check = list(base_params.keys())
                
                for p_check in all_keys_to_check:
                    true_val = current_params[p_check]
                    
                    # Logic to handle Scalar vs Array (Reusing your logic)
                    if np.isscalar(true_val):
                        # Scalar Logic
                        if p_check in summary.index:
                            results.append({
                                'varied_param': param_name, # <-- NEW: What were we stressing?
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
                        # Array Logic (Flatten and find)
                        true_val_arr = np.array(true_val)
                        it = np.nditer(true_val_arr, flags=['multi_index'])
                        for x in it:
                            idx_tuple = it.multi_index
                            idx_str_raw = ",".join(map(str, idx_tuple))
                            
                            # Candidate name generation (Same as your script)
                            candidates = [f"{p_check}[{idx_str_raw}]"]
                            if len(idx_tuple) == 1 and idx_tuple[0] < len(coord_groups):
                                candidates.append(f"{p_check}[{coord_groups[idx_tuple[0]]}]")
                            if len(idx_tuple) == 2:
                                g_i, e_i = idx_tuple
                                if g_i < len(coord_groups) and e_i < len(coord_entities):
                                    candidates.append(f"{p_check}[{coord_groups[g_i]}, {coord_entities[e_i]}]")
                                    candidates.append(f"{p_check}[{coord_groups[g_i]},{coord_entities[e_i]}]")

                            # Match in summary
                            for cand in candidates:
                                if cand in summary.index:
                                    results.append({
                                        'varied_param': param_name, # <-- NEW
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
                results.append({
                    'varied_param': param_name,
                    'step_index': step_i,
                    'param_measured': 'CRASH',
                    'est_mean': np.nan,
                    'true_value': np.nan,
                    'divergences': -1
                })

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