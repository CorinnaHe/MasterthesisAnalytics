from scipy.stats import ttest_ind
from scipy.stats import kruskal
import pandas as pd
from scipy.stats import chi2_contingency

from data_loader import load_experiment_data
from thesis.figure_creation import plot_binary_rate_per_condition, plot_switching_rate, \
    plot_initial_final_per_condition, plot_initial_vs_final_agreement


def full_balance_table(df: pd.DataFrame):
    rows = []

    # Continuous
    numeric_cols = [
        "age",
        "ai_literacy_sk9",
        "ai_literacy_sk10",
        "ai_literacy_ail2",
        "ai_literacy_ue2",
        "risk_aversion"
    ]

    for var in numeric_cols:
        row = {"variable": var}

        groups = []
        for cond in ["C1", "C2", "C3"]:
            data = df[df["condition"] == cond][var].dropna()
            row[cond] = f"{data.mean():.2f} ({data.std():.2f})"
            groups.append(data)

        stat, p = kruskal(*groups)
        row["test"] = f"H={stat:.2f}"
        row["p"] = f"{p:.4f}"

        rows.append(row)

    # Categorical
    categorical_vars = ["gender", "education", "domain_experience"]

    for var in categorical_vars:
        contingency = pd.crosstab(df[var], df["condition"])
        percent = contingency.div(contingency.sum(axis=0), axis=1) * 100

        chi2, p, dof, _ = chi2_contingency(contingency)

        for category in contingency.index:
            row = {"variable": f"{var}: {category}"}

            for cond in ["C1", "C2", "C3"]:
                count = contingency.loc[category, cond]
                perc = percent.loc[category, cond]
                row[cond] = f"{count} ({perc:.1f}%)"

            row["test"] = f"Chi2={chi2:.2f}"
            row["p"] = f"{p:.4f}"

            rows.append(row)

    return pd.DataFrame(rows)

def print_statistics(df: pd.DataFrame):
    numeric_cols = [
        "age",
        "ai_literacy_sk9",
        "ai_literacy_sk10",
        "ai_literacy_ail2",
        "ai_literacy_ue2",
        "risk_aversion"
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

    print("=== Across Conditions ===")
    with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None
    ):
        print(full_balance_table(df))


    for var in numeric_cols:
        groups = [
            df[df["condition"] == "C1"][var],
            df[df["condition"] == "C2"][var],
            df[df["condition"] == "C3"][var]
        ]

        stat, p = kruskal(*groups)
        print(f"{var}: H={stat:.3f}, p={p:.3f}")

    categorical_vars = ["gender", "education", "domain_experience"]

    for var in categorical_vars:
        contingency = pd.crosstab(df[var], df["condition"])
        chi2, p, dof, _ = chi2_contingency(contingency)

        print(f"{var}: chi2={chi2:.3f}, p={p:.3f}")


if __name__ == '__main__':
    experiment_date = "2026-03-20"
    (
        main_trials_df,
        control_measures_df,
        participants_df,
        *_
    ) = load_experiment_data(f"all_apps_wide-{experiment_date}.csv")

    condition_df = (
        main_trials_df
        .groupby('participant_code')['condition']
        .first()  # or another aggregation if needed
        .reset_index()
    )

    control_measures_df = control_measures_df.merge(
        condition_df,
        on='participant_code',
        how='left'
    )

    participant_stats = participants_df.merge(
        control_measures_df,
        left_on="participant_code",
        right_on="participant_code"
    )

    mismatch_df = main_trials_df[
        main_trials_df["initial_agree_ai"] == 0
        ].copy()

    top1_mismatch_df = main_trials_df[
        main_trials_df["initial_top_1_agree"] == 0
        ].copy()

    print("=== Participants and Trials per Condition ===")
    summary = participants_df.groupby('condition')['participant_code'].describe()
    summary['trials'] = summary['count'] * 15
    print(summary)

    print("=== Accuracy Descriptives ===")
    print("> Initial Accuracy by Condition")
    print(main_trials_df.groupby('condition')['initial_correct'].describe())
    print("> Final Accuracy by Condition")
    print(main_trials_df.groupby('condition')['final_correct'].describe())
    #plot_binary_rate_per_condition(main_trials_df, column="final_correct", y_label="Accuracy")

    print("=== Switched Descriptives ===")
    print("> Switched by Condition (full df)")
    print(main_trials_df.groupby('condition')['switched'].describe())
    #plot_binary_rate_per_condition(main_trials_df, column="switched", y_label="Switched Rate")

    print("> Switching Rate by Top_1 AI Agree")
    print(main_trials_df.groupby('initial_top_1_agree')['switched'].describe())
    print("> Switching Rate by Top_1 AI Agree by Condition")
    print(main_trials_df.groupby(['condition', 'initial_top_1_agree'])['switched'].describe())
    plot_switching_rate(
        df=main_trials_df,
        group_col="initial_top_1_agree",
        switch_col="switched",
        x_label="Initial Human–AI Match with Top-Ranked Position",
        y_label="Switching Rate (%)"
    )

    for condition, df_cond in main_trials_df.groupby('condition'):
        print(f"\n--- Condition: {condition} ---")

        plot_switching_rate(
            df=df_cond,
            group_col="initial_top_1_agree",
            switch_col="switched",
            x_label="Initial Human–AI Match",
            y_label="Switch (%)",
        )

    print("> Switched by Condition (top 1 mismatch df)")
    print(top1_mismatch_df.groupby('condition')['switched'].describe())


    print("> Switching Rate by full AI Agree")
    print(main_trials_df.groupby('initial_agree_ai')['switched'].describe())
    print("> Switching Rate by full AI Agree by Condition")
    print(main_trials_df.groupby(['condition', 'initial_agree_ai'])['switched'].describe())

    print("> Switched by Condition (complete mismatch df)")
    print(mismatch_df.groupby('condition')['switched'].describe())


    print("=== Initial Agreement Rates ===")
    print(main_trials_df.groupby('condition')['initial_agree_ai'].describe())

    print("=== Final Agreement Rates ===")
    print(main_trials_df.groupby('condition')['final_agree_ai'].describe())

    #plot_initial_vs_final_agreement(main_trials_df)

    print("=== Human Confidence===")
    print("> Initial Confidence by Condition")
    print(main_trials_df.groupby('condition')['initial_confidence'].describe())
    print("> Final Confidence by Condition")
    print(main_trials_df.groupby('condition')['final_confidence'].describe())

    print("=== AI Confidence===")
    print("> by Condition")
    print(main_trials_df.groupby('condition')['shared_ai_confidence'].describe())

    print_statistics(control_measures_df)

    print("=== AI Attitude ===")
    print("> by Condition")
    print(control_measures_df.groupby('condition')['ai_attitude'].describe())

    print("=== AI Trust ===")
    print("> by Condition")
    print(control_measures_df.groupby('condition')['ai_trust'].describe())

    print("=== Page Times ===")
    print("> by Condition")
    print(participants_df.groupby("condition")["mean_page_duration_stage1"].describe())
    print(participants_df.groupby("condition")["mean_page_duration_stage2"].describe())
