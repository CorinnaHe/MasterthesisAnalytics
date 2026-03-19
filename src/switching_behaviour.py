import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from data_loader import load_experiment_data
from inspect_data import inspect_human_ai_match
from variable_constructer import construct_variables_df

if __name__ == '__main__':
    experiment_date = "2026-03-13"
    (
        participants_df,
        example_trials_df,
        main_trials_df,
        control_measures_df,
    ) = load_experiment_data(f"all_apps_wide-{experiment_date}.csv")
    print(main_trials_df.groupby('participant_code').describe())
    print(main_trials_df.groupby('condition')['participant_code'].nunique())
    main_trials_df["confidence_gap"] = main_trials_df["confidence_gap"].astype(float)


    # variables

    print(f"\n=== General Switching Behaviour ===")
    #inspect_human_ai_match(main_trials_df, "global")
    print(f"\n=== C1 Switching Behaviour ===")
    c1_df = main_trials_df[(main_trials_df["condition"] == "C1")]
    #inspect_human_ai_match(c1_df, "global")
    print(f"\n=== C2 Switching Behaviour ===")
    c2_df = main_trials_df[(main_trials_df["condition"] == "C2")]
    #inspect_human_ai_match(c2_df, "global")
    print(f"\n=== C3 Switching Behaviour ===")
    c3_df = main_trials_df[(main_trials_df["condition"] == "C3")]
    #inspect_human_ai_match(c3_df, "global")

    print(f"\n=== Match & Switched ===")
    switched_by_match_df = main_trials_df[
        (main_trials_df["initial_agree_ai"] == 1) &
        (main_trials_df["switched"] == 1)
        ]
    n_participants = switched_by_match_df["participant_code"].nunique()
    n_total_switches = len(switched_by_match_df)
    print("Unique participants who switched:", n_participants)
    print("Total switching trials:", n_total_switches)
    switches_per_participant = (
        switched_by_match_df
        .groupby("participant_code")
        .size()
        .describe()
    )
    print(switches_per_participant)
    case_counts = (
        switched_by_match_df
        .groupby("case_id")
        .size()
        .sort_values(ascending=False)
    )
    print(case_counts.head(20))
    condition_counts = (
        switched_by_match_df
        .groupby("condition")
        .size()
    )
    print(condition_counts)
    print(switched_by_match_df["initial_confidence"].describe())
    switched_by_match_df["confidence_change"] = (
            switched_by_match_df["final_confidence"] -
            switched_by_match_df["initial_confidence"]
    )
    print(switched_by_match_df["confidence_change"].describe())
    switched_by_match_df_C3 = switched_by_match_df[(switched_by_match_df["condition"] == "C3")]
    initial_match_count = (
            switched_by_match_df["initial_decision"]
            == switched_by_match_df["cp_set_el1"]
    ).sum()

    # Final decision matches el1
    final_match_count = (
            switched_by_match_df["final_decision"]
            == switched_by_match_df["cp_set_el1"]
    ).sum()

    print(initial_match_count, final_match_count)



    # switching nearly only happens when initial human AI mismatch
    mismatch_df = main_trials_df[
        main_trials_df["initial_agree_ai"] == 0
    ].copy()
    print(mismatch_df['switched'].value_counts(normalize=True))

    print(f"\n=== Switched by Condition ===")
    print(mismatch_df.groupby('condition')['switched'].describe())
    model = smf.logit(
        "switched ~ condition",
        data=mismatch_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": mismatch_df["participant_code"]}
    )
    print(model.summary())

    print(f"\n=== Switched by Initial Confidence ===")
    print(mismatch_df.groupby('initial_confidence')['switched'].describe())

    for condition in ["C1", "C2", "C3"]:
        condition_df = mismatch_df[mismatch_df["condition"] == condition]

        model = smf.logit(
            "switched ~ initial_confidence",
            data=condition_df
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": condition_df["participant_code"]}
        )
        print(f"\n===== {condition.upper()} =====")
        print(model.summary())

    print(f"\n=== Switched by AI Confidence ===")
    print(mismatch_df.groupby('shared_ai_confidence')['switched'].describe())
    print(mismatch_df.groupby(['condition', 'shared_ai_confidence'])['switched'].describe())
    print(mismatch_df.groupby(['condition', 'set_size'])['switched'].describe())

    for condition in ["C1", "C2", "C3"]:
        condition_df = mismatch_df[mismatch_df["condition"] == condition]
        print(condition_df.groupby('shared_ai_confidence')['switched'].describe())

        model = smf.logit(
            f"switched ~ shared_ai_confidence",
            data=condition_df
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": condition_df["participant_code"]}
        )
        print(f"\n===== {condition.upper()} =====")
        print(model.summary())

    print(f"\n=== Switched by Confidence Gap ===")
    print(
        main_trials_df.groupby("condition")["confidence_gap"].describe())

    for condition in ["C1", "C2", "C3"]:
        condition_df = mismatch_df[mismatch_df["condition"] == condition]

        model = smf.logit(
            f"switched ~ confidence_gap",
            data=condition_df
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": condition_df["participant_code"]}
        )
        print(f"\n===== {condition.upper()} =====")
        print(model.summary())

    print(f"\n=== Switched by Confidence Gap + Initial Confidence ===")
    for condition in ["C1", "C2", "C3"]:
        condition_df = mismatch_df[mismatch_df["condition"] == condition]

        model = smf.logit(
            f"switched ~ confidence_gap + initial_confidence",
            data=condition_df
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": condition_df["participant_code"]}
        )
        print(f"\n===== {condition.upper()} =====")
        print(model.summary())

    print(f"\n=== Switched by Confidence Gap * Condition ===")
    model = smf.logit(
        f"switched ~ confidence_gap * C(condition)",
        data=mismatch_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": mismatch_df["participant_code"]}
    )
    print(model.summary())

    print(pd.crosstab(
        mismatch_df["condition"],
        mismatch_df["switched"]
    ))
    print(pd.crosstab(
        mismatch_df["condition"],
        mismatch_df["confidence_gap"]
    ))
    print(pd.crosstab(
        mismatch_df["confidence_gap"],
        mismatch_df["switched"]
    ))

    print(f"\n=== Switched by Confidence Gap * Condition + Initial Confidence ===")
    model = smf.logit(
        "switched ~ confidence_gap * C(condition) + initial_confidence",
        data=mismatch_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": mismatch_df["participant_code"]}
    )
    print(model.summary())

    print(f"\n=== Switched by AI Confidence + Initial Confidence===")
    for condition in ["C1", "C2", "C3"]:
        condition_df = mismatch_df[mismatch_df["condition"] == condition]

        model = smf.logit(
            f"switched ~ shared_ai_confidence + initial_confidence",
            data=condition_df
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": condition_df["participant_code"]}
        )
        print(f"\n===== {condition.upper()} =====")
        print(model.summary())