import numpy as np


def compute_cc_categories_initial(
        df,
        participant_col,
        confidence_col="initial_confidence",
        correct_col="initial_correct",
        final_correct_col="final_correct",
):
    """
    Compute Confidence–Correctness (C–C) categories per trial
    using per-participant median split (initial stage).

    Keeps final correctness for later error-rate analysis.
    """

    # Keep required columns
    data = df[
        [participant_col, confidence_col, correct_col, final_correct_col]
    ].copy()

    # --- 1️⃣ Compute per-participant median ---
    participant_medians = (
        data
        .groupby(participant_col)[confidence_col]
        .median()
        .rename("median_conf")
    )

    data = data.merge(
        participant_medians,
        left_on=participant_col,
        right_index=True
    )

    # --- 2️⃣ High vs Low classification ---
    data["high_conf"] = data[confidence_col] > data["median_conf"]

    # --- 3️⃣ C–C categories ---
    conditions = [
        (data["high_conf"] & (data[correct_col] == 1)),
        (~data["high_conf"] & (data[correct_col] == 0)),
        (data["high_conf"] & (data[correct_col] == 0)),
        (~data["high_conf"] & (data[correct_col] == 1)),
    ]

    categories = [
        "C-C Matched (High & Correct)",
        "C-C Matched (Low & Incorrect)",
        "Overconfident (High & Incorrect)",
        "Underconfident (Low & Correct)",
    ]

    data["cc_category"] = np.select(
        conditions,
        categories,
        default="Unclassified"
    )

    data["cc_matched"] = data["cc_category"].str.contains("Matched")

    return data