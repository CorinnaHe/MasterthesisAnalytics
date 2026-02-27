import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

from confidence_calibration import reliability_analysis, compute_ece_per_person, compute_cc_categories_initial
from data_loader import load_experiment_data
from variable_constructer import construct_variables_df

if __name__ == '__main__':
    experiment_date = "2026-02-25"
    (
        participants_df,
        example_trials_df,
        main_trials_df,
        control_measures_df,

    ) = load_experiment_data(f"all_apps_wide-{experiment_date}.csv")

    main_trials_df = construct_variables_df(main_trials_df)

    print(f"\n=== Trial Level ===")
    initial_reliability_metrics = reliability_analysis(
        df=main_trials_df,
        confidence_col="initial_confidence",
        correct_col="initial_correct",
        normalize_method="divide_by_max",
        discrete_bins=True,
        plot=False,
    )
    print(initial_reliability_metrics["bin_statistics"])
    print("ECE:", initial_reliability_metrics["ECE"])

    final_reliability_metrics = reliability_analysis(
        df=main_trials_df,
        confidence_col="final_confidence",
        correct_col="final_correct",
        normalize_method="divide_by_max",
        discrete_bins=True,
        plot=False,
    )
    print(final_reliability_metrics["bin_statistics"])
    print("ECE:", final_reliability_metrics["ECE"])

    ece_initial_df = compute_ece_per_person(
        df=main_trials_df,
        participant_col="participant_code",
        confidence_col="initial_confidence",
        correct_col="initial_correct"
    )

    ece_final_df = compute_ece_per_person(
        df=main_trials_df,
        participant_col="participant_code",
        confidence_col="final_confidence",
        correct_col="final_correct"
    )

    ece_combined = ece_initial_df.merge(
        ece_final_df,
        on="participant_code",
        suffixes=("_initial", "_final")
    )

    print(ece_combined.head())
    print(ece_combined.describe())

    print(f"\n=== Instance Level ===")
    cc_df = compute_cc_categories_initial(
        df=main_trials_df,
        participant_col="participant_code",
        confidence_col="initial_confidence",
        correct_col="initial_correct"
    )

    print(cc_df.head(5))
    print(cc_df.groupby("cc_category").describe())

    error_by_cc = (
        cc_df
        .groupby("cc_matched")["final_correct"]
        .agg(["count", "mean"])
        .reset_index()
    )
    # Error rate = 1 - accuracy
    error_by_cc["error_rate"] = 1 - error_by_cc["mean"]
    print(error_by_cc)

    per_person_error = (
        cc_df
        .groupby(["participant_code", "cc_matched"])["final_correct"]
        .mean()
        .reset_index()
    )

    per_person_error["error_rate"] = 1 - per_person_error["final_correct"]
    print(per_person_error.head())

    pivot_error = per_person_error.pivot(
        index="participant_code",
        columns="cc_matched",
        values="error_rate"
    ).reset_index()

    pivot_error.columns = ["participant_code", "error_mismatched", "error_matched"]
    print(pivot_error.head())


    stat, p = wilcoxon(
        pivot_error["error_mismatched"],
        pivot_error["error_matched"]
    )

    print("Wilcoxon statistic:", stat)
    print("p-value:", p)

    plt.boxplot([
        pivot_error["error_matched"],
        pivot_error["error_mismatched"]
    ])
    plt.xticks([1, 2], ["Matched", "Mismatched"])
    plt.ylabel("Final Error Rate")
    plt.show()

    print(f"\nInitial Correctness & Confidence")
    print(main_trials_df["initial_correct"].mean())
    print(main_trials_df.groupby("initial_correct")["initial_confidence"].mean())

    model = smf.logit(
        "initial_correct ~ initial_confidence",
        data=main_trials_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": main_trials_df["participant_code"]}
    )
    print(model.summary())

    print(main_trials_df["initial_correct"].mean())

    print(f"\nFinal Correctness & Confidence")
    print(main_trials_df["final_correct"].mean())
    print(main_trials_df.groupby("final_correct")["final_confidence"].mean())

    model = smf.logit(
        "final_correct ~ final_confidence",
        data=main_trials_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": main_trials_df["participant_code"]}
    )
    print(model.summary())

    print(f"\n Interaktionseffekt Initial Human-AI-Mismatch")
    model_inter = smf.logit(
        "final_correct ~ final_confidence * initial_agree_ai",
        data=main_trials_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": main_trials_df["participant_code"]}
    )
    print(model_inter.summary())

    beta_conf = model_inter.params["final_confidence"]
    beta_inter = model_inter.params["final_confidence:initial_agree_ai[T.1]"]

    print(model_inter.params.index)
    print("Effect in Mismatch (agree=0):", beta_conf)
    print("Effect in Match (agree=1):", beta_conf + beta_inter)

    print("OR Mismatch:", np.exp(beta_conf))
    print("OR Match:", np.exp(beta_conf + beta_inter))

    print(f"\n=== Arda ===")
    print(f"\nInitial Correctness & Confidence")
    mismatch_df = main_trials_df[
        main_trials_df["initial_agree_ai"] == 0
    ].copy()
    print(mismatch_df.groupby("initial_correct")["initial_confidence"].mean())


    model = smf.logit(
        "initial_correct ~ initial_confidence",
        data=mismatch_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": mismatch_df["participant_code"]}
    )
    print(model.summary())

    print(mismatch_df["initial_correct"].mean())

    print(f"\nFinal Correctness & Confidence")
    print(mismatch_df["final_correct"].mean())
    print(mismatch_df.groupby("final_correct")["final_confidence"].mean())

    model = smf.logit(
        "final_correct ~ final_confidence",
        data=mismatch_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": mismatch_df["participant_code"]}
    )
    print(model.summary())

    print(f"\n=== Confidence Calibration within Groups ===")
    print(len(mismatch_df))
    print(mismatch_df["participant_code"].isna().sum())
    print(mismatch_df.isna().sum())

    for g in ["C1", "C2", "C3"]:
        df_g = mismatch_df[mismatch_df["condition"] == g]

        model = smf.logit(
            "final_correct ~ final_confidence",
            data=df_g
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": df_g["participant_code"]}
        )

        print(f"\nGruppe {g}")
        print(model.summary())

    interaction_model = smf.logit(
        "final_correct ~ final_confidence * C(condition)",
        data=mismatch_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": mismatch_df["participant_code"]}
    )

    print(interaction_model.summary())

    print(f"\n=== Confidence Calibration over time ===")

    # Gibt es eine Accuracy Trend?
    print(main_trials_df.groupby("trial_index")["initial_correct"].mean())

    # Confidence Calibration over Time
    model_time = smf.logit(
        "initial_correct ~ initial_confidence * trial_index",
        data=main_trials_df
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": main_trials_df["participant_code"]}
    )
    print(model_time.summary())

    main_trials_df["trial_block_3"] = pd.cut(
        main_trials_df["trial_index"],
        bins=[0, 5, 10, 15],
        labels=["early", "mid", "late"],
        include_lowest=True
    )
    for block in ["early", "mid", "late"]:
        block_df = main_trials_df[main_trials_df["trial_block_3"] == block]

        model = smf.logit(
            "initial_correct ~ initial_confidence",
            data=block_df
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": block_df["participant_code"]}
        )

        print(f"\n===== {block.upper()} =====")
        print(model.summary())