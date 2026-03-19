from data_loader import load_experiment_data

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


if __name__ == '__main__':
    experiment_date = "2026-03-13"
    (
        main_trials_df,
        control_measures_df,
        *_

    ) = load_experiment_data(f"all_apps_wide-{experiment_date}.csv")
    main_trials_df = main_trials_df.merge(
        control_measures_df,
        left_on="participant_code",
        right_on="participant_code"
    )

    # Yin et al. 4.1
    accuracy_by_condition = main_trials_df.groupby("condition")["final_correct"].mean()
    print(accuracy_by_condition)

    print(main_trials_df[["final_confidence_norm", "final_calibration_score"]].describe())

    calibration_by_condition = main_trials_df.groupby("condition")["final_calibration_score"].mean()

    print(calibration_by_condition)

    # Plot
    bins = np.linspace(0, 1, 11)
    main_trials_df["confidence_bin"] = pd.cut(
        main_trials_df["final_confidence_norm"],
        bins=bins,
        include_lowest=True
    )
    calibration_data = (
        main_trials_df.groupby(["condition", "confidence_bin"])
        .agg(
            mean_confidence=("final_confidence_norm", "mean"),
            accuracy=("final_correct", "mean"),
            count=("final_correct", "size")
        )
        .reset_index()
    )
    plt.figure(figsize=(8, 6))

    for condition in calibration_data["condition"].unique():
        subset = calibration_data[calibration_data["condition"] == condition]

        plt.plot(
            subset["mean_confidence"],
            subset["accuracy"],
            marker="o",
            label=condition
        )

    # perfect calibration line
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Calibration Curve by Condition")

    plt.legend()
    #plt.show()

    summary = (
        main_trials_df.groupby("condition")
        .agg(
            mean_final_human_accuracy=("final_correct", "mean"),
            mean_confidence=("final_confidence_norm", "mean"),
            mean_calibration=("final_calibration_score", "mean"),
            mean_ai_accuracy=("ai_correct", "mean"),
            observations=("final_correct", "size")
        )
    )
    with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None
    ):
        print(summary)

    # Yin et al. 4.2.1
    main_trials_df["C2"] = (main_trials_df["condition"] == "C2").astype(int)
    main_trials_df["C3"] = (main_trials_df["condition"] == "C3").astype(int)
    scale_cols = [
        "ai_literacy",
        "ai_attitude",
        "ai_trust",
        "risk_aversion",
        "cognitive_load_mental",
        "shared_ai_confidence"
    ]
    scaler = StandardScaler()
    main_trials_df[scale_cols] = scaler.fit_transform(main_trials_df[scale_cols])

    # accuracy
    model_accuracy = smf.mixedlm(
        "final_correct ~ C(condition) + experience + ai_literacy + ai_trust + \
         ai_correct + switched  + shared_ai_confidence",
        main_trials_df,
        groups=main_trials_df["participant_code"],
        vc_formula={"case": "0 + C(case_id)"}
    )
    result_accuracy = model_accuracy.fit(method="powell")
    print(result_accuracy.summary())

    # Yin et al. 4.2.2
    # confidence calibration
    model_calibration = smf.mixedlm(
        "final_calibration_score ~ C(condition) + experience + ai_literacy + ai_trust + \
         ai_correct + switched  + shared_ai_confidence",
        main_trials_df,
        groups=main_trials_df["participant_code"],
        vc_formula={"case": "0 + C(case_id)"}
    )
    result_calibration = model_calibration.fit(method="powell")
    print(result_calibration.summary())

    # added
    model = smf.mixedlm(
        "final_calibration_score ~ C(condition) * shared_ai_confidence + experience + ai_literacy + ai_trust + \
         ai_correct + switched",
        main_trials_df,
        groups=main_trials_df["participant_code"]
    )
    result = model.fit(method="powell")
    print(result.summary())

    c3_df = main_trials_df[(main_trials_df["condition"] == "C3")]
    model = smf.mixedlm(
        "final_calibration_score ~ initial_agree_ai",
        c3_df,
        groups=c3_df["participant_code"]
    )
    result = model.fit()
    print(result.summary())

    # 5.1 Mechanisms
    # switching behavior as mechanism
    # added only initial mismatch dataframe to Yin et al. analysis
    main_trials_df["final_agree_ai"] = main_trials_df["final_agree_ai"].astype(int)
    main_trials_df["ai_correct"] = main_trials_df["ai_correct"].astype(int)
    disagree_df = main_trials_df[
        main_trials_df["initial_agree_ai"] == 0
        ]
    model = smf.mixedlm(
        "switched ~ C(condition) \
         + experience + ai_literacy + ai_trust \
         + ai_correct + shared_ai_confidence",
        disagree_df,
        groups=disagree_df["participant_code"]
    )
    result = model.fit()
    print(result.summary())

    # AI Reliance
    model = smf.mixedlm(
        "final_agree_ai ~ C(condition) + ai_correct + shared_ai_confidence",
        disagree_df,
        groups=disagree_df["participant_code"]
    )
    print(model.fit().summary())

    # 5.3 AI Concordance with AI
    # Does the treatment help humans distinguish good AI advice from bad AI advice?
    correct_df = disagree_df[disagree_df["ai_correct"] == 1]
    incorrect_df = disagree_df[disagree_df["ai_correct"] == 0]
    model_correct = smf.mixedlm(
        "final_agree_ai ~ C(condition) + shared_ai_confidence",
        correct_df,
        groups=correct_df["participant_code"]
    )
    print(model_correct.fit().summary())
    model_incorrect = smf.mixedlm(
        "final_agree_ai ~ C(condition) + shared_ai_confidence",
        incorrect_df,
        groups=incorrect_df["participant_code"]
    )
    print(model_incorrect.fit().summary())