import pandas as pd

from figures import plot_binary_stacked_bar, plot_box_with_jitter


def _accuracy_metrics(x: pd.DataFrame) -> pd.Series:
    user_only_accuracy = (x["initial_decision"] == x["y_true"]).mean()
    team_accuracy = (x["final_decision"] == x["y_true"]).mean()

    return pd.Series({
        "user_only_accuracy": user_only_accuracy,
        "team_accuracy": team_accuracy,
        "team_delta": team_accuracy - user_only_accuracy,
        "ai_accuracy": x["ai_correct"].mean(),
        "n_trials": len(x)
    })


def inspect_accuracy(trials: pd.DataFrame, label: str):
    print(f"\n=== Accuracy inspection: {label} ===")
    accuracy_per_participant = (
        trials.groupby("participant_code")
        .apply(_accuracy_metrics)
        .reset_index()
    )
    print("\nPer participant")
    print(accuracy_per_participant)

    accuracy_per_case = (
        trials.groupby("case_id")
        .apply(_accuracy_metrics)
        .reset_index()
    )
    print("\nPer case")
    print(accuracy_per_case)

    accuracy_per_trial = (
        trials.groupby("trial_index")
        .apply(_accuracy_metrics)
        .reset_index()
    )
    print("\nPer trial")
    print(accuracy_per_trial)

    accuracy_per_case = (
        trials
        .assign(y_true_col=trials["y_true"])
        .groupby("y_true")
        .apply(lambda x: _accuracy_metrics(x.assign(y_true=x["y_true_col"])))
        .reset_index()
    )
    print("\nPer class")
    print(accuracy_per_case)

    global_accuracy = (
        accuracy_per_participant
        [["user_only_accuracy", "team_accuracy", "ai_accuracy"]]
        .describe()
    )
    print("\nGlobal accuracy")
    print(global_accuracy)


def inspect_human_ai_match(trials: pd.DataFrame, label: str):
    print(f"\n=== Human AI match inspection: {label} ===")

    # ------------------------------------------------------------------
    # 1. Define decision labels ONCE (paper-aligned)
    # ------------------------------------------------------------------
    label_map = {
        (1, 1): "Match\nSwitch",
        (1, 0): "Match\nNot Switch",
        (0, 1): "Mismatch\nSwitch",
        (0, 0): "Mismatch\nNot Switch",
    }

    trials["decision_label"] = trials.apply(
        lambda r: label_map[(r["initial_agree_ai"], r["switched"])],
        axis=1
    )

    # ------------------------------------------------------------------
    # 2. Switching behaviour (Figure 5c logic)
    # ------------------------------------------------------------------
    switch_by_match = (
        trials
        .groupby("initial_agree_ai")["switched"]
        .value_counts()
        .unstack(fill_value=0)
    )

    print("\nSwitching behaviour by initial Human–AI match")
    print(switch_by_match)

    switch_rates = switch_by_match.div(switch_by_match.sum(axis=1), axis=0)
    print("\nSwitching rates")
    print(switch_rates)

    # ------------------------------------------------------------------
    # 3. Counts dict for stacked bar plot (derived from labels)
    # ------------------------------------------------------------------
    counts = (
        trials
        .groupby("decision_label")
        .size()
        .to_dict()
    )

    stacked_counts = {
        "Match": {
            "Not Switch": counts.get("Match\nNot Switch", 0),
            "Switch": counts.get("Match\nSwitch", 0),
        },
        "Mismatch": {
            "Not Switch": counts.get("Mismatch\nNot Switch", 0),
            "Switch": counts.get("Mismatch\nSwitch", 0),
        },
    }

    plot_binary_stacked_bar(
        stacked_counts,
        outcome_order=["Not Switch", "Switch"],
        colors={
            "Not Switch": "#cccccc",
            "Switch": "#4daf4a",
        },
        ylabel="Switch",
        xlabel="Initial Human–AI Match",
    )

    # ------------------------------------------------------------------
    # 4. Confidence descriptives (paper Table / Fig. 5d logic)
    # ------------------------------------------------------------------
    confidence_summary = (
        trials
        .groupby("decision_label")["delta_confidence"]
        .describe()
    )

    print("\nConfidence by decision choice")
    print(confidence_summary)

    # ------------------------------------------------------------------
    # 5. Box + jitter plot (Figure 5d)
    # ------------------------------------------------------------------
    plot_box_with_jitter(
        df=trials,
        x_col="decision_label",
        y_col="delta_confidence",
        order=[
            "Match\nNot Switch",
            "Mismatch\nSwitch",
            "Mismatch\nNot Switch",
        ],
    )


def inspect_h2(trials: pd.DataFrame):
    print(trials.groupby("condition")["appropriate_reliance"].mean())