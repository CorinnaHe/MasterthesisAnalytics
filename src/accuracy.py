from scipy.stats import shapiro, kruskal
import statsmodels.formula.api as smf

from data_loader import load_experiment_data
import scikit_posthocs as sp


def cliffs_delta(x, y):
    n1 = len(x)
    n2 = len(y)

    greater = sum(i > j for i in x for j in y)
    less = sum(i < j for i in x for j in y)

    delta = (greater - less) / (n1 * n2)
    return delta


if __name__ == '__main__':
    experiment_date = "2026-03-20"
    (
        main_trials_df,
        control_measures_df,
        participant_stats,
        *_

    ) = load_experiment_data(f"all_apps_wide-{experiment_date}.csv")
    main_trials_df = main_trials_df.merge(
        control_measures_df,
        left_on="participant_code",
        right_on="participant_code"
    )

    # Schauer paper
    print(participant_stats.head())

    summary = participant_stats.groupby("condition")["final_accuracy"].agg(["mean", "std", "count"])
    print(summary)

    # Shapiro-Wilk (Normalverteilungs-Test)
    stat, p = shapiro(participant_stats["final_accuracy"])
    print("Shapiro-Wilk p-value:", p) # p < 0.05 nicht normalverteilt, wie auch bei Schauer

    # Kruskal-Wallis Test
    groups = [
        participant_stats[participant_stats.condition == "C1"]["final_accuracy"],
        participant_stats[participant_stats.condition == "C2"]["final_accuracy"],
        participant_stats[participant_stats.condition == "C3"]["final_accuracy"]
    ]

    H, p = kruskal(*groups)

    n = len(participant_stats)
    k = 3

    eta2 = (H - k + 1) / (n - k)

    print("H:", H)
    print("p:", p)
    print("eta^2:", eta2) # small-moderate effect
    # p < 0.05 -> signifikanter Effekt der Conditions auf Accuracy (wie bei Schauer, wobei die ja andere Conditions haben)

    # Post-hoc Dunn Test
    dunn = sp.posthoc_dunn(
        participant_stats,
        val_col="final_accuracy",
        group_col="condition",
        p_adjust="fdr_bh"  # Benjamini-Hochberg
    )
    print(dunn)
    # C1 vs. C2 nur marginal, C1 vs. C3 signifikant. C2 vs. C3 nicht signifikant

    # Pairwise tests Cliffs Delta
    c1 = participant_stats[participant_stats.condition == "C1"]["final_accuracy"]
    c2 = participant_stats[participant_stats.condition == "C2"]["final_accuracy"]
    c3 = participant_stats[participant_stats.condition == "C3"]["final_accuracy"]

    print("C1 vs C2:", cliffs_delta(c1, c2)) # small-medium
    print("C1 vs C3:", cliffs_delta(c1, c3)) # medium
    print("C2 vs C3:", cliffs_delta(c2, c3)) # /

    # Nicht mehr Schauer :D
    print("=== Logistic Regression correct ~ condition ===")
    model = smf.logit(
        "final_correct ~ C(condition)",
        data=main_trials_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": main_trials_df["participant_code"]}
    )
    print(model.summary())

    print("=== Logistic Regression correct ~ condition + case_id ===")
    model = smf.logit(
        "final_correct ~ C(condition) + C(case_id)",
        data=main_trials_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": main_trials_df["participant_code"]}
    )
    print(model.summary())

    print("=== Logistic Regression correct ~ condition + case_id ===")
    model = smf.logit(
        "final_correct ~ C(condition) + C(case_id)",
        data=main_trials_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": main_trials_df["participant_code"]}
    )
    print(model.summary())

    print("=== A3: Logistic Regression with interaction correct ~ condition * ai_correct + case_id ===")
    model = smf.logit(
        "final_correct ~ C(condition) * ai_correct + C(case_id)",
        data=main_trials_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": main_trials_df["participant_code"]}
    )
    print(model.summary())

    print("=== A4: Logistic Regression with interaction correct ~ condition * shared_ai_confidence ===")
    model = smf.logit(
        "final_correct ~ C(condition) * shared_ai_confidence",
        data=main_trials_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": main_trials_df["participant_code"]}
    )
    print(model.summary())

    print("=== A7: Roboustness Check Case_id ===")
    model = smf.logit(
        "final_correct ~ C(condition) * shared_ai_confidence + C(case_id)",
        data=main_trials_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": main_trials_df["participant_code"]}
    )
    print(model.summary())

    conditions = main_trials_df["condition"].unique()
    for cond in conditions:
        df_cond = main_trials_df[main_trials_df["condition"] == cond]
        print(f"=== A5: Logistic Regression {cond} only ~ shared_ai_confidence * ai_correct ===")

        model = smf.logit(
            "final_correct ~ shared_ai_confidence * ai_correct",
            data=df_cond
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": df_cond["participant_code"]}
        )
        print(model.summary())

    print("=== Robustness Checks with Controls ===")
    formula = """
        final_correct ~
        C(condition) +
        age +
        C(gender) +
        C(education) +
        ai_literacy_sk9 +
        ai_literacy_sk10 +
        ai_literacy_ail2 +
        ai_literacy_ue2 +
        C(domain_experience) +
        risk_aversion +
        cognitive_load_mental +
        C(case_id)
    """
    model = smf.logit(
        formula,
        data=main_trials_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": main_trials_df["participant_code"]}
    )
    print(model.summary())


    print("=== Does Switching Drives Accuracy ===")
    mismatch_df = main_trials_df[
        main_trials_df["initial_top_1_agree"] == 0
        ].copy()

    switch_model = smf.logit(
        "final_correct ~ switched * C(condition)",
        data=mismatch_df
    ).fit(cov_type="cluster",
          cov_kwds={"groups": mismatch_df["participant_code"]})
    print(switch_model.summary())

    switch_model = smf.logit(
        "final_correct ~ switched * ai_correct + C(condition)",
        data=mismatch_df
    ).fit(cov_type="cluster",
          cov_kwds={"groups": mismatch_df["participant_code"]})
    print(switch_model.summary())


    model = smf.ols(
        "final_correct ~ switched * top1_correct + C(condition)",
        data=mismatch_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": mismatch_df["participant_code"]}
    )
    print(model.summary())

    switch_model = smf.logit(
        "final_correct ~ switched_to_top1 * top1_correct + C(condition)",
        data=mismatch_df
    ).fit(cov_type="cluster",
          cov_kwds={"groups": mismatch_df["participant_code"]})
    print(switch_model.summary())

    model = smf.ols(
        "final_correct ~ switched_to_top1 * top1_correct + C(condition)",
        data=mismatch_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": mismatch_df["participant_code"]}
    )
    print(model.summary())

    print("=== Does Reliance Drives Accuracy ===")
    mismatch_df = main_trials_df[
        main_trials_df["initial_top_1_agree"] == 0
        ].copy()

    switch_model = smf.logit(
        "final_correct ~ appropriate_reliance * C(condition)",
        data=mismatch_df
    ).fit(cov_type="cluster",
          cov_kwds={"groups": mismatch_df["participant_code"]})
    print(switch_model.summary())
