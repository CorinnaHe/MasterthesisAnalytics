import pandas as pd

def _construct_reliance_metrics(trials: pd.DataFrame) -> pd.DataFrame:
    # AI correctness
    trials["set_valid"] = (
        (trials["y_true"].eq("poor") & trials["cp_contains_poor"]) |
        (trials["y_true"].eq("standard") & trials["cp_contains_standard"]) |
        (trials["y_true"].eq("good") & trials["cp_contains_good"])
    )

    trials["pp_correct"] = trials["point_pred_cal"].eq(trials["y_true"])

    # human–AI agreement
    trials["initial_agree_pp"] = trials["initial_decision"].eq(trials["point_pred_cal"])
    trials["final_agree_pp"] = trials["final_decision"].eq(trials["point_pred_cal"])

    trials["initial_agree_set"] = (
        (trials["initial_decision"].eq("good") & trials["cp_contains_good"]) |
        (trials["initial_decision"].eq("standard") & trials["cp_contains_standard"]) |
        (trials["initial_decision"].eq("poor") & trials["cp_contains_poor"])
    )

    trials["final_agree_set"] = (
        (trials["final_decision"].eq("good") & trials["cp_contains_good"]) |
        (trials["final_decision"].eq("standard") & trials["cp_contains_standard"]) |
        (trials["final_decision"].eq("poor") & trials["cp_contains_poor"])
    )

    # switching behavior
    trials["switched_to_pp"] = ~trials["initial_agree_pp"] & trials["final_agree_pp"]
    trials["switched_to_set"] = ~trials["initial_agree_set"] & trials["final_agree_set"]

    # reliance
    trials["over_reliance_pp"] = trials["final_agree_pp"] & ~trials["pp_correct"]
    trials["under_reliance_pp"] = ~trials["final_agree_pp"] & trials["pp_correct"]

    trials["over_reliance_set"] = trials["switched_to_set"] & ~trials["set_valid"]
    trials["under_reliance_set"] = ~trials["final_agree_set"] & trials["set_valid"]

    return trials


def _construct_confidence_metrics(trials: pd.DataFrame) -> pd.DataFrame:
    trials["delta_confidence"] = (
        trials["final_confidence"] - trials["initial_confidence"]
    ).astype(int)

    trials["final_correct"] = trials["final_decision"].eq(trials["y_true"])

    return trials


def construct_variables_df(trials: pd.DataFrame) -> pd.DataFrame:
    trials = _construct_reliance_metrics(trials)
    trials = _construct_confidence_metrics(trials)
    return trials
