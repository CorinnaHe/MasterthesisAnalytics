import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency

from data_loader import load_experiment_data
from inspect_data import inspect_human_ai_match

if __name__ == '__main__':
    experiment_date = "2026-03-20"
    (
        main_trials_df,
        *_
    ) = load_experiment_data(f"all_apps_wide-{experiment_date}.csv")
    print(main_trials_df.groupby('participant_code').describe())
    print(main_trials_df.groupby('condition')['participant_code'].nunique())
    main_trials_df["confidence_gap"] = main_trials_df["confidence_gap"].astype(float)

    main_trials_df['ai_not_low'] = (main_trials_df['shared_ai_norm'] > 0).astype(int)


    # variables
    print(f"\n=== General Switching Behaviour ===")
    inspect_human_ai_match(main_trials_df, "global")
    print(f"\n=== C1 Switching Behaviour ===")
    c1_df = main_trials_df[(main_trials_df["condition"] == "C1")]
    #inspect_human_ai_match(c1_df, "global")
    print(f"\n=== C2 Switching Behaviour ===")
    c2_df = main_trials_df[(main_trials_df["condition"] == "C2")]
    #inspect_human_ai_match(c2_df, "global")
    print(f"\n=== C3 Switching Behaviour ===")
    c3_df = main_trials_df[(main_trials_df["condition"] == "C3")]
    #inspect_human_ai_match(c3_df, "global")

    print(f"\n=== Inspect Page Times ===")
    print(main_trials_df.groupby("condition")["mean_page_duration"].agg(["mean", "std", "count"]))
    print(main_trials_df.groupby("switched")["mean_page_duration"].agg(["mean", "std", "count"]))
    print(main_trials_df.groupby("initial_agree_ai")["mean_page_duration"].agg(["mean", "std", "count"]))
    print(main_trials_df.groupby(["initial_agree_ai", "switched"])["mean_page_duration"].agg(["mean", "std", "count"]))

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
    #print(mismatch_df['switched'].value_counts(normalize=True))
    #mismatch_df = mismatch_df[mismatch_df["shared_ai_confidence"].isin([2, 3])].copy()
    mismatch_df= main_trials_df[
        main_trials_df["initial_top_1_agree"] == 0
    ].copy()

    #mismatch_df = mismatch_df[mismatch_df["condition"].isin(["C2", "C1"])].copy()
    print(mismatch_df[['initial_confidence_norm', 'shared_ai_norm']].describe())

    print("=== AI Confidence matters more in C3??? ===")
    model = smf.logit(
        f"switched ~ initial_confidence_norm + shared_ai_confidence * C(condition)",
        data=mismatch_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": mismatch_df["participant_code"]}
    )
    print(model.summary())

    model = smf.logit(
        f"switched ~ shared_ai_confidence * C(condition)",
        data=mismatch_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": mismatch_df["participant_code"]}
    )
    print(model.summary())

    print(f"\n=== Switched by AI Confidence + Initial Confidence ===")

    model = smf.logit(
        f"switched ~ confidence_gap * C(condition)",
        data=mismatch_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": mismatch_df["participant_code"]}
    )
    print(model.summary())

    model = smf.logit(
        f"switched ~ confidence_gap + I(confidence_gap**2) * C(condition)",
        data=mismatch_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": mismatch_df["participant_code"]}
    )
    print(model.summary())

    model = smf.logit(
        f"switched ~ C(initial_confidence_norm) * C(ai_not_low)",
        data=mismatch_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": mismatch_df["participant_code"]}
    )
    print(model.summary())

    model = smf.logit(
        f"switched ~ ai_not_low * C(condition)",
        data=mismatch_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": mismatch_df["participant_code"]}
    )
    print(model.summary())

    model = smf.logit(
        f"switched ~ initial_confidence_norm * C(condition)",
        data=mismatch_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": mismatch_df["participant_code"]}
    )
    print(model.summary())

    print(
        pd.crosstab(
            [mismatch_df["condition"], mismatch_df["shared_ai_confidence"]],
            mismatch_df["switched"]
        )
    )

    model = smf.logit(
        f"switched ~ initial_confidence_norm + shared_ai_confidence",
        data=mismatch_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": mismatch_df["participant_code"]}
    )
    print(model.summary())

    model = smf.logit(
        f"switched ~ initial_confidence_norm + shared_ai_confidence * C(condition)",
        data=mismatch_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": mismatch_df["participant_code"]}
    )
    print(model.summary())


    model = smf.logit(
        f"switched ~ initial_confidence * shared_ai_confidence * C(condition)",
        data=mismatch_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": mismatch_df["participant_code"]}
    )
    print(model.summary())

    model = smf.logit(
        f"switched ~ initial_confidence * C(condition) + shared_ai_confidence * C(condition)",
        data=mismatch_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": mismatch_df["participant_code"]}
    )
    print(model.summary())



    print(f"\n=== A1: Switched by Condition ===")
    print(mismatch_df.groupby('condition')['switched'].describe())
    model = smf.logit(
        "switched ~ condition",
        data=mismatch_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": mismatch_df["participant_code"]}
    )
    print(model.summary())

    print(f"\n=== A2:  "
          f"Switched by Initial Confidence ===")
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

    print(f"\n=== A3: Switched by AI Confidence ===")
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

    print(f"\n=== Switched by AI Confidence + Initial Confidence ===")
    print(model.summary())
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

    print(f"\n=== A5: Switched by Confidence Gap ===")
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

    print(f"\n=== A6a: Switched by Confidence Gap + Initial Confidence ===")
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

    print(f"\n=== A6b: Switched by Confidence Gap + AI Confidence ===")
    for condition in ["C1", "C2", "C3"]:
        condition_df = mismatch_df[mismatch_df["condition"] == condition]

        model = smf.logit(
            f"switched ~ confidence_gap + shared_ai_confidence",
            data=condition_df
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": condition_df["participant_code"]}
        )
        print(f"\n===== {condition.upper()} =====")
        print(model.summary())

    print(f"\n=== A6c: Switched by Initial Confidence + AI Confidence ===")
    for condition in ["C1", "C2", "C3"]:
        condition_df = mismatch_df[mismatch_df["condition"] == condition]

        model = smf.logit(
            f"switched ~ initial_confidence + shared_ai_confidence",
            data=condition_df
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": condition_df["participant_code"]}
        )
        print(f"\n===== {condition.upper()} =====")
        print(model.summary())

    print(f"\n=== Switched by shared_ai_confidence * Condition ===")
    model = smf.logit(
        f"switched ~ shared_ai_confidence * C(condition)",
        data=mismatch_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": mismatch_df["participant_code"]}
    )
    print(model.summary())

    print(f"\n=== Switched by initial_confidence * Condition ===")
    model = smf.logit(
        f"switched ~ initial_confidence * C(condition)",
        data=mismatch_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": mismatch_df["participant_code"]}
    )
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

    print(f"\n=== Switched by Confidence Gap + AI Correct ===")
    for condition in ["C1", "C2", "C3"]:
        condition_df = mismatch_df[mismatch_df["condition"] == condition]

        model = smf.logit(
            f"switched ~ confidence_gap + ai_correct",
            data=condition_df
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": condition_df["participant_code"]}
        )
        print(f"\n===== {condition.upper()} =====")
        print(model.summary())

    print(f"\n=== A8:  Switched by AI Correct ===")
    print(mismatch_df.groupby(['condition', 'ai_correct'])['switched'].mean())
    for condition in ["C1", "C2", "C3"]:
        condition_df = mismatch_df[mismatch_df["condition"] == condition]
        model = smf.logit(
            f"switched ~ ai_correct",
            data=condition_df
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": condition_df["participant_code"]}
        )
        print(f"\n===== {condition.upper()} =====")
        print(model.summary())

        model = smf.logit(
            f"switched ~ ai_correct + shared_ai_confidence",
            data=condition_df
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": condition_df["participant_code"]}
        )
        print(f"\n===== {condition.upper()} =====")
        print(condition_df.groupby(["shared_ai_confidence", "ai_correct"])["switched"].mean())
        print(model.summary())

    print(f"\n=== A11: Switched by Confidence Gap * Condition + Initial Confidence ===")
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

    print(f"\n=== Switched by Page Duration ===")
    mismatch_df["duration_sq"] = mismatch_df["mean_page_duration"] ** 2
    mismatch_df["log_duration"] = np.log(mismatch_df["mean_page_duration"] + 1)
    mismatch_df["log_duration_stage2"] = np.log(mismatch_df["page_duration_stage2"] + 1)
    for condition in ["C1", "C2", "C3"]:
        condition_df = mismatch_df[mismatch_df["condition"] == condition]

        model = smf.logit(
            "switched ~ mean_page_duration + duration_sq + log_duration "
            "+ confidence_gap ",
            data=condition_df
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": condition_df["participant_code"]}
        )
        print(f"\n===== {condition.upper()} =====")
        print(model.summary())

        model_stage2_full = smf.logit(
            "switched ~ page_duration_stage2 + log_duration_stage2 + confidence_gap",
            data=mismatch_df
        ).fit()
        print(model_stage2_full.summary())

        model_a = smf.ols(
            "log_duration ~ confidence_gap",
            data=mismatch_df
        ).fit()
        print(model_a.summary())

    print("=== Switched to what ===")
    switched_by_disagree = mismatch_df[
        mismatch_df["switched"] == 1
        ].copy()
    print(switched_by_disagree.groupby("condition")["final_agree_ai"].mean())
    print(switched_by_disagree.groupby(["condition", "shared_ai_confidence"])["final_agree_ai"].mean())


    c3_disagree = mismatch_df[mismatch_df["condition"] == "C3"]
    print(c3_df.groupby("initial_agree_ai")["initial_pos_in_set"].value_counts(dropna=False))
    model = smf.logit(
        "switched ~  C(initial_pos_in_set) + shared_ai_confidence + initial_confidence + ai_correct",
        data=c3_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": c3_df["participant_code"]}
    )
    print(model.summary())

    c3_switched = main_trials_df[
        (main_trials_df["switched"] == 1) & (main_trials_df["condition"] == "C3")
        ].copy()
    print(c3_switched.groupby("shared_ai_confidence")["final_agree_ai"].mean())

    c3_switched["final_choice_type"] = pd.Categorical(
        c3_switched["final_choice_type"],
        categories=["top1", "alternative", "outside"]
    )

    c3_switched["final_choice_type_code"] = c3_switched["final_choice_type"].cat.codes
    print(c3_switched.groupby("shared_ai_confidence")["final_choice_type"].value_counts(normalize=True))
    print(c3_switched["initial_pos_in_set"].value_counts(normalize=True) * 100)
    print(c3_switched["final_pos_in_set"].value_counts(normalize=True) * 100)
    print(pd.crosstab(
        c3_switched["initial_pos_in_set"],
        c3_switched["final_pos_in_set"]
    ))

    c3_switched["moved_to_top1"] = (c3_switched["final_pos_in_set"] == 1).astype(int)
    model = smf.logit(
"moved_to_top1 ~ C(initial_pos_in_set)",
        data=c3_switched
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": c3_switched["participant_code"]}
    )
    print(model.summary())

    print(print(c3_switched.groupby(["ai_correct", "final_choice_type"]).size()))
    model = smf.mnlogit(
        "final_choice_type_code ~ C(initial_pos_in_set) + shared_ai_confidence",
        data=c3_switched
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": c3_switched["participant_code"]}
    )
    print(model.summary())

    c3_switched["ignored_set"] = c3_switched["final_pos_in_set"].isna().astype(int)
    print(c3_switched.groupby("set_size")["ignored_set"].mean())

    # Does initial position affect switching?
    c3_df = main_trials_df[(main_trials_df["condition"] == "C3")].copy()
    model = smf.logit(
        "switched ~ C(initial_pos_in_set) + initial_confidence + ai_correct + shared_ai_confidence",
        data=c3_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": c3_df["participant_code"]}
    )
    print(model.summary())

    # Does initial position affect moving to top1?
    table_counts = pd.crosstab(
        c3_switched["initial_pos_in_set"],
        c3_switched["moved_to_top1"]
    )
    table_counts.columns = ["not_moved_to_top1", "moved_to_top1"]
    table_counts["total"] = table_counts.sum(axis=1)
    table_counts["share_moved_to_top1"] = table_counts["moved_to_top1"] / table_counts["total"]
    print("=== Descriptive Table ===")
    with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None
    ):
        print(table_counts)



    chi2, p, dof, expected = chi2_contingency(table_counts[["not_moved_to_top1", "moved_to_top1"]])

    print("\n=== Chi-square test ===")
    print(f"Chi2 statistic: {chi2:.4f}")
    print(f"p-value: {p:.6f}")
    print(f"Degrees of freedom: {dof}")

    model = smf.logit(
        "switched ~ initial_confidence + shared_ai_confidence",
        data=c3_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": c3_df["participant_code"]}
    )
    print(model.summary())

    model = smf.logit(
        "switched ~ ai_correct",
        data=c3_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": c3_df["participant_code"]}
    )
    print(model.summary())

    print("=== C3: Switched based on initial position ===")
    c3_top1_mismatch = mismatch_df[mismatch_df["condition"] == "C3"]
    c3_top1_mismatch["moved_to_top1"] = (c3_top1_mismatch["final_pos_in_set"] == 1).astype(int)

    print(c3_top1_mismatch.groupby("initial_pos_in_set")["switched"].describe())
    print(c3_df.groupby("initial_pos_in_set")["switched"].describe())

    model = smf.logit(
        "switched ~ C(initial_pos_in_set)",
        data=c3_top1_mismatch
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": c3_top1_mismatch["participant_code"]}
    )
    print(model.summary())

    model = smf.logit(
        "switched ~ C(initial_pos_in_set) + shared_ai_confidence + initial_confidence",
        data=c3_top1_mismatch
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": c3_top1_mismatch["participant_code"]}
    )
    print(model.summary())

    model = smf.logit(
        "switched ~ C(initial_pos_in_set) + confidence_gap + initial_confidence",
        data=c3_top1_mismatch
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": c3_top1_mismatch["participant_code"]}
    )
    print(model.summary())

    model = smf.logit(
        "switched ~ C(initial_pos_in_set) + initial_confidence",
        data=c3_top1_mismatch
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": c3_top1_mismatch["participant_code"]}
    )
    print(model.summary())

    model = smf.logit(
        "switched ~ C(initial_pos_in_set) + confidence_gap",
        data=c3_top1_mismatch
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": c3_top1_mismatch["participant_code"]}
    )
    print(model.summary())

    model = smf.logit(
        "switched ~ C(initial_pos_in_set) + shared_ai_confidence",
        data=c3_top1_mismatch
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": c3_top1_mismatch["participant_code"]}
    )
    print(model.summary())

    import pandas as pd

    mismatch_df['gap_bin'] = pd.cut(mismatch_df['confidence_gap'], bins=10)
    plot_df = mismatch_df.groupby('gap_bin')['switched'].mean().reset_index()
    plot_df['gap_mid'] = plot_df['gap_bin'].apply(lambda x: x.mid)
    import matplotlib.pyplot as plt

    plt.figure()

    plt.plot(plot_df['gap_mid'], plot_df['switched'], marker='o')

    plt.xlabel('Confidence Gap (AI - Own)')
    plt.ylabel('Probability of Switching')
    plt.title('Switching Behavior vs Confidence Gap')

    plt.axvline(0)  # key reference: AI = own confidence

    #plt.show()

    import pandas as pd

    heatmap_df = mismatch_df.groupby(
        ['initial_confidence_norm', 'shared_ai_norm']
    )['switched'].mean().reset_index()
    import pandas as pd

    heatmap_pivot = heatmap_df.pivot(
        index='initial_confidence_norm',
        columns='shared_ai_norm',
        values='switched'
    )

    import matplotlib.pyplot as plt

    plt.figure()

    plt.imshow(heatmap_pivot, aspect='auto')

    plt.colorbar(label='Probability of Switching')

    plt.xticks(range(len(heatmap_pivot.columns)), heatmap_pivot.columns)
    plt.yticks(range(len(heatmap_pivot.index)), heatmap_pivot.index)

    plt.xlabel('AI Confidence')
    plt.ylabel('Initial Confidence')
    plt.title('Switching Probability Heatmap')

    plt.figure()

    plt.imshow(heatmap_pivot, aspect='auto')
    plt.colorbar(label='Probability of Switching')

    for i in range(len(heatmap_pivot.index)):
        for j in range(len(heatmap_pivot.columns)):
            value = heatmap_pivot.iloc[i, j]
            if pd.notna(value):
                plt.text(j, i, f"{value:.2f}", ha='center', va='center')

    plt.xticks(range(len(heatmap_pivot.columns)), heatmap_pivot.columns)
    plt.yticks(range(len(heatmap_pivot.index)), heatmap_pivot.index)

    plt.xlabel('AI Confidence')
    plt.ylabel('Initial Confidence')
    plt.title('Switching Probability by Confidence Levels')

    plt.show()