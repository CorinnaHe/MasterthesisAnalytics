import numpy as np

from data_loader import load_experiment_data
from variable_constructer import construct_variables_df

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


def analyze_case_difficulty(
        df,
        difficulty_column,
        case_column="case_id",
        initial_correct_column="initial_correct",
        final_correct_column="final_correct",
        ai_correct_column="ai_correct",
        participant_column="participant_code",
        condition_column="condition",
        plot=True
):

    # --------------------------------
    # 1 Case level aggregation
    # --------------------------------

    case_df = (
        df
        .groupby(case_column)
        .agg(
            user_only_accuracy=(initial_correct_column, "mean"),
            team_accuracy=(final_correct_column, "mean"),
            ai_accuracy=(ai_correct_column, "mean"),
            difficulty_column_accuracy=(difficulty_column, "mean"),
            n_trials=(difficulty_column, "count")
        )
        .reset_index()
    )

    case_df["team_delta"] = (
        case_df["team_accuracy"]
        - case_df["user_only_accuracy"]
    )

    # --------------------------------
    # 2 Difficulty classification
    # --------------------------------

    def classify_difficulty(acc):
        if acc < 0.25:
            return "very_hard"

        if acc < 0.5:
            return "hard"

        if acc < 0.75:
            return "medium"

        else:
            return "easy"

    case_df["difficulty"] = case_df["difficulty_column_accuracy"].apply(classify_difficulty)

    # --------------------------------
    # 3 Merge difficulty back to trial-level data
    # --------------------------------

    df_analysis = df.merge(
        case_df[[case_column, "difficulty", "user_only_accuracy"]],
        on=case_column,
        how="left"
    )

    # --------------------------------
    # 4 Difficulty summary
    # --------------------------------

    difficulty_summary = (
        case_df
        .groupby("difficulty")
        .agg(
            human_accuracy=("user_only_accuracy", "mean"),
            team_accuracy=("team_accuracy", "mean"),
            improvement=("team_delta", "mean"),
            ai_accuracy=("ai_accuracy", "mean"),
            n_cases=(case_column, "count")
        )
        .reset_index()
        .sort_values("human_accuracy")
    )

    with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None
    ):
        print("\n=== CASE LEVEL DATA ===")
        print(case_df.sort_values("user_only_accuracy"))

        print("\n=== DIFFICULTY SUMMARY ===")
        print(difficulty_summary)

    # --------------------------------
    # 5 Regression using ALL observations
    # --------------------------------

    print("\n=== Logistic Regression (clustered by participant) ===")

    model = smf.logit(
        f"{final_correct_column} ~ C(difficulty)",
        data=df_analysis
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": df_analysis[participant_column]}
    )

    print(model.summary())

    print("\n=== Logistic Regression with condition interaction ===")

    model2 = smf.logit(
        f"{final_correct_column} ~ C(difficulty) * C({condition_column})",
        data=df_analysis
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": df_analysis[participant_column]}
    )

    print(model2.summary())

    # --------------------------------
    # 6 Visualization (case level)
    # --------------------------------

    if plot:

        plt.figure(figsize=(7,5))

        sns.scatterplot(
            data=case_df,
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

    return case_df, difficulty_summary, df_analysis

def analyze_set_size(
        df,
        difficulty_column,
        case_column="case_id",
        initial_correct_column="initial_correct",
        final_correct_column="final_correct",
        ai_correct_column="ai_correct",
        participant_column="participant_code",
        condition_column="condition",
):

    # --------------------------------
    # 1 Case level aggregation
    # --------------------------------

    case_df = (
        df
        .groupby(case_column)
        .agg(
            user_only_accuracy=(initial_correct_column, "mean"),
            team_accuracy=(final_correct_column, "mean"),
            ai_accuracy=(ai_correct_column, "mean"),
            difficulty_column_accuracy=(difficulty_column, "mean"),
            n_trials=(difficulty_column, "count")
        )
        .reset_index()
    )

    case_df["team_delta"] = (
            case_df["team_accuracy"]
            - case_df["user_only_accuracy"]
    )

    # --------------------------------
    # 2 Difficulty classification
    # --------------------------------

    def classify_difficulty(acc):
        if acc < 0.25:
            return "very_hard"

        if acc < 0.5:
            return "hard"

        if acc < 0.75:
            return "medium"

        else:
            return "easy"

    case_df["difficulty"] = case_df["difficulty_column_accuracy"].apply(classify_difficulty)

    # --------------------------------
    # 3 Merge difficulty back to trial-level data
    # --------------------------------

    df_analysis = df.merge(
        case_df[[case_column, "difficulty", "user_only_accuracy"]],
        on=case_column,
        how="left"
    )

    cp_df = df_analysis[
        df_analysis["condition"] == "C3"
    ]

    set_size_df = (
        df_analysis
        .groupby("set_size")
        .agg(
            user_only_accuracy=(initial_correct_column, "mean"),
            team_accuracy=(final_correct_column, "mean"),
            ai_accuracy=(ai_correct_column, "mean"),
            n_trials=(case_column, "count")
        )
        .reset_index()
    )

    set_size_df["team_delta"] = (
        set_size_df["team_accuracy"]
        - set_size_df["user_only_accuracy"]
    )


    with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None
    ):
        print("\n=== SET SIZE SUMMARY ===")
        print(set_size_df)

    # --------------------------------
    # 5 Regression using ALL observations
    # --------------------------------

    print("\n=== Logistic Regression (clustered by participant) ===")

    model = smf.logit(
        f"{final_correct_column} ~ C(set_size)",
        data=cp_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": cp_df[participant_column]}
    )

    print(model.summary())

    print("\n=== Logistic Regression with condition interaction ===")

    model2 = smf.logit(
        f"{final_correct_column} ~ C(set_size) + C(difficulty)",
        data=cp_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": cp_df[participant_column]}
    )

    print(model2.summary())


def analyze_set_size_switching(
        df,
        difficulty_column,
        case_column="case_id",
        initial_correct_column="initial_correct",
        final_correct_column="final_correct",
        ai_correct_column="ai_correct",
        participant_column="participant_code",
        condition_column="condition",
):
    # --------------------------------
    # 1 Case level aggregation
    # --------------------------------

    case_df = (
        df
        .groupby(case_column)
        .agg(
            user_only_accuracy=(initial_correct_column, "mean"),
            team_accuracy=(final_correct_column, "mean"),
            ai_accuracy=(ai_correct_column, "mean"),
            difficulty_column_accuracy=(difficulty_column, "mean"),
            n_trials=(difficulty_column, "count")
        )
        .reset_index()
    )

    case_df["team_delta"] = (
            case_df["team_accuracy"]
            - case_df["user_only_accuracy"]
    )

    # --------------------------------
    # 2 Difficulty classification
    # --------------------------------

    def classify_difficulty(acc):
        if acc < 0.25:
            return "very_hard"

        if acc < 0.5:
            return "hard"

        if acc < 0.75:
            return "medium"

        else:
            return "easy"

    case_df["difficulty"] = case_df["difficulty_column_accuracy"].apply(classify_difficulty)

    # --------------------------------
    # 3 Merge difficulty back to trial-level data
    # --------------------------------

    df_analysis = df.merge(
        case_df[[case_column, "difficulty", "user_only_accuracy"]],
        on=case_column,
        how="left"
    )

    df_analysis["interface_type"] = np.where(
        df_analysis[condition_column] == "C3",
        "C3_size" + df_analysis["set_size"].astype(str),
        df_analysis[condition_column]
    )

    set_size_df = (
        df_analysis
        .groupby("interface_type")
        .agg(
            user_only_accuracy=(initial_correct_column, "mean"),
            team_accuracy=(final_correct_column, "mean"),
            ai_accuracy=(ai_correct_column, "mean"),
            n_trials=(case_column, "count")
        )
        .reset_index()
    )

    set_size_df["team_delta"] = (
            set_size_df["team_accuracy"]
            - set_size_df["user_only_accuracy"]
    )

    with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None
    ):
        print("\n=== INTERFACE_TYPE SUMMARY ===")
        print(set_size_df)

    # --------------------------------
    # 5 Regression using ALL observations
    # --------------------------------

    print("\n=== Logistic Regression (clustered by participant) ===")

    model = smf.logit(
        f"switched_to_ai ~ C(interface_type) + C(difficulty)",
        data=df_analysis
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": df_analysis[participant_column]}
    )

    print(model.summary())


if __name__ == '__main__':
    experiment_date = "2026-03-13"
    (
        participants_df,
        example_trials_df,
        main_trials_df,
        control_measures_df,
    ) = load_experiment_data(f"all_apps_wide-{experiment_date}.csv")

    case_df, difficulty_summary, df_analysis = analyze_case_difficulty(
        df=main_trials_df,
        difficulty_column="initial_correct",
        plot=False
    )

    case_df, difficulty_summary, df_analysis = analyze_case_difficulty(
        df=main_trials_df,
        difficulty_column="ai_correct",
        plot=False
    )

    print("\n=== INITIAL HUMAN-AI MISMATCH DF ===")
    mismatch_df = main_trials_df[
        main_trials_df["initial_agree_ai"] == 0
    ].copy()

    case_df, difficulty_summary, df_analysis = analyze_case_difficulty(
        df=mismatch_df,
        difficulty_column="initial_correct",
        plot=False
    )

    case_df, difficulty_summary, df_analysis = analyze_case_difficulty(
        df=mismatch_df,
        difficulty_column="ai_correct",
        plot=False
    )

    analyze_set_size(
        main_trials_df,
        difficulty_column="ai_correct",
    )

    analyze_set_size_switching(
        mismatch_df,
        difficulty_column="initial_correct",
    )

