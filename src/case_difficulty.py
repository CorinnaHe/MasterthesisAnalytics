import numpy as np

from data_loader import load_experiment_data

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


def analyze_case_difficulty(
        df,
        difficulty_column,
        final_correct_column="final_correct",
        participant_column="participant_code",
        condition_column="condition",
        plot=True
):
    df_model = df.copy()
    # remove rows with missing difficulty
    df_model = df_model.dropna(subset=[difficulty_column])

    print("\n=== Logistic Regression (clustered by participant) ===")
    model = smf.logit(
        f"{final_correct_column} ~  C({difficulty_column}, Treatment(reference='easy'))",
        data=df_model
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": df_model[participant_column]}
    )

    print(model.summary())

    print("\n=== Logistic Regression with condition interaction ===")

    model2 = smf.logit(
        f"{final_correct_column} ~ C({difficulty_column}, Treatment(reference='easy')) * C({condition_column})",
        data=df_model
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": df_model[participant_column]}
    )

    print(model2.summary())

    # Visualization
    if plot:
        plt.figure(figsize=(7,5))

        sns.scatterplot(
            data=df_model,
            x="difficulty_column_accuracy",
            y="team_delta",
            hue="difficulty",
            s=120
        )

        plt.axhline(0, linestyle="--")

        plt.xlabel(f"{difficulty_column} (Case Difficulty)")
        plt.ylabel("Human-AI Team Improvement")
        plt.title("Improvement vs Case Difficulty")

        plt.show()

def analyze_set_size(
        df,
        difficulty_column,
        final_correct_column="final_correct",
        participant_column="participant_code",
        condition_column="condition",
):
    df_model = df.copy()
    # remove rows with missing difficulty
    df_model = df_model.dropna(subset=[difficulty_column])
    cp_df = df_model[df_model[condition_column] == "C3"]

    set_size_df = (
        df_model
        .groupby("set_size")
        .agg(
            user_only_accuracy=("initial_correct", "mean"),
            team_accuracy=(final_correct_column, "mean"),
            ai_accuracy=("ai_accuracy", "mean"),
            n_trials=("case_id", "count")
        )
        .reset_index()
    )

    set_size_df["team_delta"] = (
        set_size_df["team_accuracy"] - set_size_df["user_only_accuracy"]
    )

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print("\n=== SET SIZE SUMMARY ===")
        print(set_size_df)

    print("\n=== Logistic Regression (clustered by participant) ===")

    model = smf.logit(
        f"{final_correct_column} ~ C(set_size)",
        data=cp_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": cp_df[participant_column]}
    )

    print(model.summary())

    print("\n=== Logistic Regression with difficulty control ===")

    model2 = smf.logit(
        f"{final_correct_column} ~ C(set_size) + C({difficulty_column}, Treatment(reference='easy'))",
        data=cp_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": cp_df[participant_column]}
    )

    print(model2.summary())


def analyze_set_size_switching(
        df,
        difficulty_column,
        participant_column="participant_code",
        condition_column="condition",
):
    df_model = df.copy()
    # remove rows with missing difficulty
    df_model = df_model.dropna(subset=[difficulty_column])

    df_model["interface_type"] = np.where(
        df_model[condition_column] == "C3",
        "C3_size" + df_model["set_size"].astype(str),
        df_model[condition_column]
    )

    set_size_df = (
        df_model
        .groupby("interface_type")
        .agg(
            user_only_accuracy=("initial_correct", "mean"),
            team_accuracy=("final_correct", "mean"),
            ai_accuracy=("ai_accuracy", "mean"),
            n_trials=("case_id", "count")
        )
        .reset_index()
    )

    set_size_df["team_delta"] = (
        set_size_df["team_accuracy"] - set_size_df["user_only_accuracy"]
    )

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print("\n=== INTERFACE_TYPE SUMMARY ===")
        print(set_size_df)

    print("\n=== Logistic Regression (clustered by participant) ===")

    model = smf.logit(
        f"switched_to_ai ~ C(interface_type) + C({difficulty_column}, Treatment(reference='easy'))",
        data=df_model
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": df[participant_column]}
    )

    print(model.summary())


def _print_statistics(case_stats, difficulty_column):
    difficulty_summary = (
        case_stats
        .groupby(difficulty_column)
        .agg(
            human_accuracy=("user_only_accuracy", "mean"),
            team_accuracy=("team_accuracy", "mean"),
            improvement=("team_delta", "mean"),
            ai_accuracy=("ai_accuracy", "mean"),
            n_cases=("case_id", "count")
        )
        .reset_index()
        .sort_values("human_accuracy")
    )

    with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None
    ):
        print("\n=== CASE LEVEL DATA ===")
        if (difficulty_column == "human_difficulty"):
            print(case_stats.sort_values("user_only_accuracy"))
        else:
            print(case_stats.sort_values("ai_accuracy"))

        print("\n=== DIFFICULTY SUMMARY ===")
        print(difficulty_summary)


if __name__ == '__main__':
    experiment_date = "2026-03-20"
    (
        main_trials_df,
        _,
        _,
        case_stats,
        *_
    ) = load_experiment_data(f"all_apps_wide-{experiment_date}.csv")

    human_column = "human_difficulty"
    ai_column = "ai_difficulty"
    print("\n=== Human difficulty ===")
    _print_statistics(case_stats, human_column)
    print("\n=== ai difficulty ===")
    _print_statistics(case_stats, ai_column)

    df = main_trials_df.merge(
        case_stats[
            [
                "case_id",
                "user_only_accuracy",
                "team_accuracy",
                "ai_accuracy",
                "team_delta",
                human_column,
                ai_column,
            ]
        ],
        on="case_id",
        how="left"
    )

    analyze_case_difficulty(df, "human_difficulty", plot=False)
    #analyze_case_difficulty(df, "ai_difficulty", plot=False) # quasi perfect seperation between easy and harder

    mismatch_df = df[df["initial_agree_ai"] == 0].copy()

    analyze_case_difficulty(mismatch_df, "human_difficulty", plot=False)
    #analyze_case_difficulty(mismatch_df, "ai_difficulty", plot=False)

    #analyze_set_size(df, difficulty_column="ai_difficulty")
    analyze_set_size_switching(mismatch_df, difficulty_column="human_difficulty")
