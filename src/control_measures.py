import pandas as pd

from data_loader import load_experiment_data
from variable_constructer import construct_variables_df


def print_statistics(df: pd.DataFrame):
    numeric_cols = [
        "age",
        "ai_literacy_sk9",
        "ai_literacy_sk10",
        "ai_literacy_ail2",
        "ai_literacy_ue2",
        "ai_attitude",
        "ai_trust",
        "risk_aversion",
        "domain_experience"
    ]

    with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None
    ):
        print(df[numeric_cols].describe())
    print(df["gender"].value_counts(dropna=False))
    print(df["education"].value_counts(dropna=False))


if __name__ == '__main__':
    experiment_date = "2026-03-10"
    (
        participants_df,
        example_trials_df,
        main_trials_df,
        control_measures_df,

    ) = load_experiment_data(f"all_apps_wide-{experiment_date}.csv")
    main_trials_df = construct_variables_df(main_trials_df)

    print_statistics(control_measures_df)
