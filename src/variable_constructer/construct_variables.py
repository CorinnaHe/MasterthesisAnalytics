import numpy as np
import pandas as pd

def _construct_reliance_metrics(trials: pd.DataFrame) -> pd.DataFrame:
    trials = trials.copy()

    mask_pp = trials["condition"].isin(["C1", "C2"])
    mask_set = trials["condition"] == "C3"

    trials["is_set_based"] = (trials["condition"] == "C3").astype(int)
    # human correctness
    trials["initial_correct"] = (
            (trials["y_true"].eq("poor") & trials["initial_decision"]) |
            (trials["y_true"].eq("standard") & trials["initial_decision"]) |
            (trials["y_true"].eq("good") & trials["initial_decision"])
    ).astype(int)

    trials["final_correct"] = (
            (trials["y_true"].eq("poor") & trials["final_decision"]) |
            (trials["y_true"].eq("standard") & trials["final_decision"]) |
            (trials["y_true"].eq("good") & trials["final_decision"])
    ).astype(int)

    # ai correctness
    trials["point_pred_correct"] = (
        trials["point_pred_cal"]
        .eq(trials["y_true"])
    ).astype(int)

    trials["set_based_correct"] = (
            (trials["y_true"].eq("poor") & trials["cp_contains_poor"]) |
            (trials["y_true"].eq("standard") & trials["cp_contains_standard"]) |
            (trials["y_true"].eq("good") & trials["cp_contains_good"])
    ).astype(int)

    trials["ai_correct"] = np.where(
        trials["condition"].isin(["C1", "C2"]),
        trials["point_pred_correct"],
        trials["set_based_correct"]
    )

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

    # set_size
    cp_cols = [
        "cp_contains_poor",
        "cp_contains_standard",
        "cp_contains_good"
    ]

    #trials["set_size"] = np.nan
    #trials.loc[mask_set, "set_size"] = (
    #    trials.loc[mask_set, cp_cols]
    #    .astype(int)
    #    .sum(axis=1)
    #)
    trials["set_size"] = (
        trials[cp_cols]
        .astype(int)
        .sum(axis=1)
    )


    # shared_ai_confidence: common 1–3 scale (high = 3) across point- and set-based conditions.
    #   For point-based (C1, included in mask_pp), use point_pred_confidence thresholds:
    #   low=1 if <0.625, medium=2 if 0.625–0.84, high=3 if >0.84.
    #   For set-based (mask_set), invert set_size so 1->3, 2->2, 3->1 to align with the same scale.
    trials["shared_ai_confidence"] = pd.NA
    conf = trials.loc[mask_pp, "point_pred_confidence"]
    trials.loc[mask_pp, "shared_ai_confidence"] = (
            (conf > 0.84).astype(int) * 3
            + ((conf >= 0.625) & (conf <= 0.84)).astype(int) * 2
            + (conf < 0.625).astype(int) * 1
    )
    trials.loc[mask_set, "shared_ai_confidence"] = trials.loc[mask_set, "set_size"].map({1: 3, 2: 2, 3: 1})

    # human-ai confidence gap: shared ai confidence norm - initial human confidence norm
    trials["initial_confidence_norm"] = (trials["initial_confidence"] - 1) / 4
    trials["shared_ai_norm"] = (trials["shared_ai_confidence"] - 1) / 2
    trials["confidence_gap"] = trials["shared_ai_norm"] - trials["initial_confidence_norm"]    # signed gap

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
    trials["initial_correct"] = trials["initial_decision"].eq(trials["y_true"]).astype(int)

    return trials


def construct_variables_df(trials: pd.DataFrame) -> pd.DataFrame:
    trials = _construct_reliance_metrics(trials)
    trials = _construct_confidence_metrics(trials)
    return trials
