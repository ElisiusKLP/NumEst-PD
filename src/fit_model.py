from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import pymc as pm
import matplotlib.pyplot as plt
from datetime import datetime
from src.model import build_subject_model

def get_paths():
    """
    Return a list of all file paths of csv data
    """
    dir = "data"
    base = Path.cwd()
    data_dir = base / dir
    data_files = list(data_dir.glob('*.csv'))
    print(f"data_files {data_files}")
    return data_files

def get_subject_data():
    files = get_paths()
    file = files[0]
    df = pd.read_csv(file)
    print(df.head(5))
    subject_df = df[df["S_ID"] == "S_00"]

    n_stim = subject_df["Presented_numerosity"]
    y_obs = subject_df["Estimated_numerosity"]

    print(f"Length n_stim {len(n_stim)}")
    print(f"n_stim: {n_stim.value_counts().sort_index()}")
    print(f"Length y_obs {len(y_obs)}")
    print(f"y_obs: {y_obs.value_counts().sort_index()}")
    return n_stim, y_obs

def main():
    n_stim, y_obs = get_subject_data()

    # convert from pandas series to numpy arrays
    n_stim = n_stim.values.astype(float)
    y_obs  = y_obs.values.astype(float)

    print(f"Building model ...")
    model = build_subject_model(n_stim, y_obs)

    sample_config = {
        "draws": 2000,
        "tune": 2000,
        "target_accept": 0.9,
        "chains": 4
    }

    # Sampling
    with model:
        result = pm.sample(
            **sample_config
        )
    
    result

    output_obj = {
        "config": sample_config,
        "trace": result
    }

    # Current datetime as string
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"sample_{dt_str}.joblib"
    print(f"Saving Trace to {filename}")

    joblib.dump(result, filename)
    

if __name__ == "__main__":
    main()