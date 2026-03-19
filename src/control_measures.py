import numpy as np
import pandas as pd

from data_loader import load_experiment_data
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind

def print_statistics(df: pd.DataFrame):
    numeric_cols = [
        "age",
        "cognitive_load_mental",
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

    gender_counts = df["gender"].value_counts(dropna=False)
    gender_percent = df["gender"].value_counts(dropna=False, normalize=True) * 100
    gender_summary = pd.DataFrame({
        "count": gender_counts,
        "percent": gender_percent
    })
    print(gender_summary)

    education_counts = df["education"].value_counts(dropna=False)
    education_percent = df["education"].value_counts(dropna=False, normalize=True) * 100
    education_summary = pd.DataFrame({
        "count": education_counts,
        "percent": education_percent
    })
    print(education_summary)

    domain_experience_counts = df["domain_experience"].value_counts(dropna=False)
    domain_experience_percent = df["domain_experience"].value_counts(dropna=False, normalize=True) * 100
    domain_experience_summary = pd.DataFrame({
        "count": domain_experience_counts,
        "percent": domain_experience_percent
    })
    print(domain_experience_summary)


def analyze_mental_load(
        main_trials_df,
        control_measures_df,
        mental_load_column="cognitive_load_mental",
        switched_column="switched",
        initial_correct_column="initial_correct",
        final_correct_column="final_correct",
        participant_column="participant_code",
        condition_column="condition",
):
    df = main_trials_df.merge(
        control_measures_df,
        left_on=participant_column,
        right_on=participant_column
    )

    participant_df = (
        df
        .groupby(participant_column)
        .agg(
            condition=(condition_column, "first"),
            mental_load=(mental_load_column, "first"),
            initial_accuracy=(initial_correct_column, "mean"),
            final_accuracy=(final_correct_column, "mean"),
            switch_rate=(switched_column, "mean"),
            n_trials=(initial_correct_column, "count")
        )
        .reset_index()
    )

    with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None
    ):
        print("\n=== CONDITION SUMMARY ===")
        print(
            df
            .groupby(condition_column)
            .agg(
                mental_load=(mental_load_column, "mean"),
                n_trials=(participant_column, "count")
            )
            .reset_index()
        )

    print("\n=== Condition Logistic Regression ===")
    model = smf.ols(
        "mental_load ~ C(condition)",
        data=participant_df
    ).fit()
    print(model.summary())
    anova = sm.stats.anova_lm(model, typ=2)
    print(anova)

    # Welch
    c1_load = participant_df[participant_df["condition"] == "C1"]["mental_load"]
    c3_load = participant_df[participant_df["condition"] == "C3"]["mental_load"]
    t_stat, p_value = ttest_ind(c1_load, c3_load, equal_var=False)
    print("T-statistic:", t_stat)
    print("p-value:", p_value)

    # Cohen
    mean1 = np.mean(c1_load)
    mean2 = np.mean(c3_load)
    std1 = np.std(c1_load, ddof=1)
    std2 = np.std(c3_load, ddof=1)
    n1 = len(c1_load)
    n2 = len(c3_load)
    pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / pooled_std
    print("Cohen's d:", cohens_d)

    print("\n=== Mental Load -> Final Accuracy ===")
    model2 = smf.ols(
        "final_accuracy ~ mental_load",
        data=participant_df
    ).fit()
    print(model2.summary())

    print("\n=== Mental Load -> Switch Rate ===")
    print(
        participant_df
        .groupby("switch_rate")
        .agg(
            mental_load=("mental_load", "mean"),
            n_trials=(participant_column, "count")
        )
        .reset_index()
    )
    model3 = smf.ols(
        "switch_rate ~ mental_load",
        data=participant_df
    ).fit()
    print(model3.summary())

    print("\n=== Combined Model ===")
    model4 = smf.ols(
        "switch_rate ~ mental_load + C(condition)",
        data=participant_df
    ).fit()
    print(model4.summary())


if __name__ == '__main__':
    experiment_date = "2026-03-13"
    (
        main_trials_df,
        control_measures_df,
        *_

    ) = load_experiment_data(f"all_apps_wide-{experiment_date}.csv")

    # Verteilung Control Measures & Mental Load
    print_statistics(control_measures_df)


    # Mental Load
    analyze_mental_load(main_trials_df, control_measures_df)


