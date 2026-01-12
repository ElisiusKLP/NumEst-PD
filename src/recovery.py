import numpy as np
import pymc as pm
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az


def simulate_hierarchical_data(
    n_subjects=20,
    n_stim_minmax=[5, 8],
    n_reps=5,
    true_params=None
):
    """Simulate data for a hierarchical LogNormal power-law model."""
    rng = np.random.default_rng(42)

    # Assign subjects to groups (0=NH, 1=PH)
    s_param_ph = np.zeros(n_subjects, dtype=int)
    s_param_ph[n_subjects // 2:] = 1

    stim_levels = range(n_stim_minmax[0], n_stim_minmax[1] + 1)
    entities = [0, 1]

    # Build trial table
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

    subject_idx = df['subject_id'].values
    group_idx_per_trial = df['group_id'].values
    entity_idx = df['entity'].values
    n_stim_val = df['n_stim'].values

    if true_params is None:
        raise ValueError("No true parameters provided.")

    # Sample subject-level parameters per entity
    log_alpha_subj = np.zeros((n_subjects, 2))
    beta_subj = np.zeros((n_subjects, 2))

    for s in range(n_subjects):
        g = s_param_ph[s]
        for e in [0, 1]:
            log_alpha_subj[s, e] = rng.normal(
                true_params['log_alpha_mu'][g, e],
                true_params['log_alpha_sigma'][g, e]
            )
            beta_subj[s, e] = rng.normal(
                true_params['beta_mu'][g, e],
                true_params['beta_sigma'][g, e]
            )

    # Select parameters per trial
    trial_log_alpha = log_alpha_subj[subject_idx, entity_idx]
    trial_beta = beta_subj[subject_idx, entity_idx]

    # Log-space mean
    mu_hat_log = trial_log_alpha + trial_beta * np.log(n_stim_val)

    # Group-specific noise
    trial_sigma = true_params['sigma_group'][group_idx_per_trial]

    # Generate observations
    y_latent = rng.lognormal(mean=mu_hat_log, sigma=trial_sigma)
    y_obs = np.round(y_latent)

    data_dict = {
        'n_stim': n_stim_val,
        'y_obs': y_obs,
        'subject_id': subject_idx,
        's_param_ph': s_param_ph,
        'entity': entity_idx,
        'dose': None
    }

    return data_dict, true_params


def run_sensitivity_analysis(model_builder, base_params, sensitivity_config, sim_kwargs):
    """One-factor-at-a-time sensitivity analysis."""
    results = []

    coord_groups = ["NH", "PH"]
    coord_entities = ["object", "human"]

    print("Starting Sensitivity Analysis...")
    print(f"Base Parameters keys: {list(base_params.keys())}")

    # for each parameter in config, simulate data and fit model
    for param_name, value_list in sensitivity_config.items():
        print(f"\n>>> TESTING PARAMETER: {param_name}")

        for step_i, test_value in enumerate(value_list):
            print(f"   Step {step_i + 1}/{len(value_list)}")

            current_params = base_params.copy()
            current_params[param_name] = test_value

            sim_data, _ = simulate_hierarchical_data(
                n_subjects=sim_kwargs.get("n_subjects", 40),
                n_stim_minmax=sim_kwargs.get("n_stim_minmax", [5, 8]),
                n_reps=sim_kwargs.get("n_reps", 5),
                true_params=current_params
            )

            try:
                model = model_builder(**sim_data)

                with model:
                    idata = pm.sample(
                        sim_kwargs.get("samples", 500),
                        tune=sim_kwargs.get("tune", 500),
                        chains=2,
                        progressbar=True
                    )

                n_div = int(idata.sample_stats.diverging.sum())
                summary = az.summary(idata, hdi_prob=0.95)

                for p_check in base_params.keys():
                    true_val = current_params[p_check]

                    if np.isscalar(true_val):
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
                        true_val_arr = np.array(true_val)
                        it = np.nditer(true_val_arr, flags=['multi_index'])

                        for x in it:
                            idx_tuple = it.multi_index
                            candidates = []

                            idx_str_raw = ",".join(map(str, idx_tuple))
                            candidates.append(f"{p_check}[{idx_str_raw}]")

                            if len(idx_tuple) == 2:
                                g_i, e_i = idx_tuple
                                if g_i < len(coord_groups) and e_i < len(coord_entities):
                                    g_lbl = coord_groups[g_i]
                                    e_lbl = coord_entities[e_i]
                                    candidates += [
                                        f"{p_check}[{g_lbl}, {e_lbl}]",
                                        f"{p_check}[{g_lbl},{e_lbl}]"
                                    ]

                            elif len(idx_tuple) == 1:
                                if idx_tuple[0] < len(coord_groups):
                                    candidates.append(f"{p_check}[{coord_groups[idx_tuple[0]]}]")

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


def plot_sensitivity(df_results, out_dir):
    """Plot recovery curves for each varied parameter."""
    varied_params = df_results['varied_param'].unique()

    # for each of the varied parameters create a plot
    for v_param in varied_params:
        subset = df_results[
            (df_results['varied_param'] == v_param) &
            (df_results['param_measured'] == v_param)
        ].copy()

        if subset.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))

        indices = subset['index_name'].unique()
        if len(indices) == 0 or (len(indices) == 1 and indices[0] is None):
            subset['index_name'] = 'Scalar'
            indices = ['Scalar']

        for idx_name in indices:
            sub_idx = subset[subset['index_name'] == idx_name]

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

        all_vals = np.concatenate([subset['true_value'], subset['est_mean']])
        min_v, max_v = all_vals.min(), all_vals.max()
        pad = (max_v - min_v) * 0.1
        ax.plot(
            [min_v - pad, max_v + pad],
            [min_v - pad, max_v + pad],
            'k--',
            alpha=0.3,
            label="Ideal"
        )

        ax.set_title(f"Sensitivity Analysis: Varying {v_param}")
        ax.set_xlabel("True Value")
        ax.set_ylabel("Estimated Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

        total_div = subset['divergences'].max()
        if total_div > 0:
            plt.figtext(0.15, 0.85, f"Max Divergences: {total_div}", color='red', weight='bold')

        plt.tight_layout()
        plt.savefig(out_dir / f"{v_param}_recov.png")
        plt.show()
