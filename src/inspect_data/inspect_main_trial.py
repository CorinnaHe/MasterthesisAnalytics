import pandas as pd


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


def inspect_h2(trials: pd.DataFrame):
    print(trials.groupby("condition")["appropriate_reliance"].mean())