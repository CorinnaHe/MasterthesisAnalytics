import pandas as pd

def _construct_reliance_metrics(trials: pd.DataFrame) -> pd.DataFrame:
    trials = trials.copy()

    mask_pp = trials["condition"].isin(["C1", "C2"])
    mask_set = trials["condition"] == "C3"

    trials["is_set_based"] = (trials["condition"] == "C3").astype(int)
    # ai correctness
    trials["ai_correct"] = pd.NA

    trials.loc[mask_pp, "ai_correct"] = (
        trials.loc[mask_pp, "point_pred_cal"]
        .eq(trials.loc[mask_pp, "y_true"])
    ).astype(int)

    trials.loc[mask_set, "ai_correct"] = (
        (trials.loc[mask_set, "y_true"].eq("poor") &
         trials.loc[mask_set, "cp_contains_poor"]) |
        (trials.loc[mask_set, "y_true"].eq("standard") &
         trials.loc[mask_set, "cp_contains_standard"]) |
        (trials.loc[mask_set, "y_true"].eq("good") &
         trials.loc[mask_set, "cp_contains_good"])
    ).astype(int)

    # human-ai agreement
    trials["initial_agree_ai"] = pd.NA
    trials["final_agree_ai"] = pd.NA

    trials.loc[mask_pp, "initial_agree_ai"] = (
        trials.loc[mask_pp, "initial_decision"]
        .eq(trials.loc[mask_pp, "point_pred_cal"])
    ).astype(int)

    trials.loc[mask_pp, "final_agree_ai"] = (
        trials.loc[mask_pp, "final_decision"]
        .eq(trials.loc[mask_pp, "point_pred_cal"])
    ).astype(int)

    trials.loc[mask_set, "initial_agree_ai"] = (
        (trials.loc[mask_set, "initial_decision"].eq("poor") &
         trials.loc[mask_set, "cp_contains_poor"]) |
        (trials.loc[mask_set, "initial_decision"].eq("standard") &
         trials.loc[mask_set, "cp_contains_standard"]) |
        (trials.loc[mask_set, "initial_decision"].eq("good") &
         trials.loc[mask_set, "cp_contains_good"])
    ).astype(int)

    trials.loc[mask_set, "final_agree_ai"] = (
        (trials.loc[mask_set, "final_decision"].eq("poor") &
         trials.loc[mask_set, "cp_contains_poor"]) |
        (trials.loc[mask_set, "final_decision"].eq("standard") &
         trials.loc[mask_set, "cp_contains_standard"]) |
        (trials.loc[mask_set, "final_decision"].eq("good") &
         trials.loc[mask_set, "cp_contains_good"])
    ).astype(int)

    # switching behavior
    trials["switched"] = (
            trials["initial_decision"] != trials["final_decision"]
    ).astype(int)

    trials["switched_to_ai"] = (
        ~trials["initial_agree_ai"] & trials["final_agree_ai"]
    ).astype(int)

    # reliance
    trials["over_reliance"] = (
        trials["final_agree_ai"] & ~trials["ai_correct"]
    ).astype(int)

    trials["under_reliance"] = (
        ~trials["final_agree_ai"] & trials["ai_correct"]
    ).astype(int)

    trials["final_agree_ai"] = trials["final_agree_ai"].astype(bool)
    trials["ai_correct"] = trials["ai_correct"].astype(bool)

    trials["appropriate_reliance"] = (
        trials["final_agree_ai"] == trials["ai_correct"]
    ).astype(int)

    return trials


def _construct_confidence_metrics(trials: pd.DataFrame) -> pd.DataFrame:
    trials["delta_confidence"] = (
        trials["final_confidence"] - trials["initial_confidence"]
    ).astype(int)

    trials["final_correct"] = trials["final_decision"].eq(trials["y_true"]).astype(int)

    return trials


def construct_variables_df(trials: pd.DataFrame) -> pd.DataFrame:
    trials = _construct_reliance_metrics(trials)
    trials = _construct_confidence_metrics(trials)
    return trials
