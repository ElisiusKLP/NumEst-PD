
import pandas as pd
from pathlib import Path
from src.utils import get_paths



def main():

    paths = get_paths("data/study2", "*ph.csv")
    print(paths)
    medpath = get_paths("data/study2", "*medication.csv")
    med_df = pd.read_csv(medpath)
    new_cols = {med_df.columns[1]: 'dose'}
    med_df = med_df.rename(columns=new_cols)
    print(f"med_df cols: {med_df.columns}")

    dfs_list = []
    for path in paths:
        df = pd.read_csv(path)
        entity = "object" if "NEO" in path.name else "human"
        df["entity"] = entity
        print(f"entity col: {df["entity"][0:5]}")

        # Merge to assign dose
        df = df.merge(
            med_df[["s_param_id_code", "dose"]],
            on="s_param_id_code",
            how="left"  # Use 'left' to keep all rows in df; NaN if no match
        )

        # verify merge worked
        print(f"Added dose values (first 5): {df['dose'].iloc[:5].tolist()}")

        dfs_list.append(df)

    df_comb = pd.concat(dfs_list, ignore_index=True)

    print(df_comb.head(10))
    print(f"length of df {len(df_comb)}")

    # check whether subjects have both object and human condition
    # Group by subject ID and collect unique entity values
    subject_entities = df_comb.groupby('S_ID')['entity'].agg(lambda x: set(x.unique()))

    # Find subjects with exactly one entity type (i.e., either 'object' and 'human')
    inconsistent_subjects = subject_entities[subject_entities.apply(lambda s: len(s) == 1)]

    if not inconsistent_subjects.empty:
        print("\n⚠️ Warning: The following subjects appear with BOTH 'object' and 'human' entities:")
        for subj_id, entities in inconsistent_subjects.items():
            print(f"  Subject {subj_id}: {entities}")
    else:
        print("\n✅ All subjects have consistent entity labels.")

    # recode the values
    


    outdir = Path("dataset")
    outdir.mkdir(exist_ok=True)
    fpath = outdir / "study2.csv"
    df_comb.to_csv(fpath, index=False)


    


if __name__ == "__main__":
    main()