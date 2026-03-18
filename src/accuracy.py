from scipy.stats import shapiro, kruskal
import statsmodels.formula.api as smf

from data_loader import load_experiment_data
from variable_constructer import construct_variables_df
import scikit_posthocs as sp


def cliffs_delta(x, y):
    n1 = len(x)
    n2 = len(y)

    greater = sum(i > j for i in x for j in y)
    less = sum(i < j for i in x for j in y)

    delta = (greater - less) / (n1 * n2)
    return delta


if __name__ == '__main__':
    experiment_date = "2026-03-13"
    (
        participants_df,
        example_trials_df,
        main_trials_df,
        control_measures_df,

    ) = load_experiment_data(f"all_apps_wide-{experiment_date}.csv")
    main_trials_df = construct_variables_df(main_trials_df)
    main_trials_df = main_trials_df.merge(
        control_measures_df,
        left_on="participant_code",
        right_on="participant_code"
    )


    # Schauer paper
    acc_df = (
        main_trials_df.groupby(["participant_code", "condition"])["final_correct"]
        .mean()
        .reset_index(name="accuracy")
    )
    print(acc_df.head())

    summary = acc_df.groupby("condition")["accuracy"].agg(["mean", "std", "count"])
    print(summary)

    # Shapiro-Wilk (Normalverteilungs-Test)
    stat, p = shapiro(acc_df["accuracy"])
    print("Shapiro-Wilk p-value:", p) # p < 0.05 nicht normalverteilt, wie auch bei Schauer

    # Kruskal-Wallis Test
    groups = [
        acc_df[acc_df.condition == "C1"]["accuracy"],
        acc_df[acc_df.condition == "C2"]["accuracy"],
        acc_df[acc_df.condition == "C3"]["accuracy"]
    ]

    H, p = kruskal(*groups)

    n = len(acc_df)
    k = 3

    eta2 = (H - k + 1) / (n - k)

    print("H:", H)
    print("p:", p)
    print("eta^2:", eta2) # small-moderate effect
    # p < 0.05 -> signifikanter Effekt der Conditions auf Accuracy (wie bei Schauer, wobei die ja andere Conditions haben)

    # Post-hoc Dunn Test
    dunn = sp.posthoc_dunn(
        acc_df,
        val_col="accuracy",
        group_col="condition",
        p_adjust="fdr_bh"  # Benjamini-Hochberg
    )
    print(dunn)
    # C1 vs. C2 nur marginal, C1 vs. C3 signifikant. C2 vs. C3 nicht signifikant

    # Pairwise tests Cliffs Delta
    c1 = acc_df[acc_df.condition == "C1"]["accuracy"]
    c2 = acc_df[acc_df.condition == "C2"]["accuracy"]
    c3 = acc_df[acc_df.condition == "C3"]["accuracy"]

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

    print("=== Logistic Regression with interaction correct ~ condition * ai_correct + case_id ===")
    model = smf.logit(
        "final_correct ~ C(condition) * ai_correct + C(case_id)",
        data=main_trials_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": main_trials_df["participant_code"]}
    )
    print(model.summary())

    print("=== Logistic Regression with interaction correct ~ condition * point_pred_confidence ===")
    model = smf.logit(
        "final_correct ~ C(condition) * point_pred_confidence",
        data=main_trials_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": main_trials_df["participant_code"]}
    )
    print(model.summary())

    print("=== Logistic Regression C3 only ~ set_size + ai_correct ===")
    c3_df = main_trials_df[
            main_trials_df["condition"] == "C3"
        ].copy()

    model = smf.logit(
        "final_correct ~ set_size + ai_correct",
        data= c3_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": c3_df["participant_code"]}
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



