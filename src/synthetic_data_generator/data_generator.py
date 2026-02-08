import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import string

from config import RAW_DATA_DIR

# -----------------------
# Configuration
# -----------------------
N_PARTICIPANTS = 150
N_EXAMPLE_TRIALS = 3
N_MAIN_TRIALS = 15
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# -----------------------
# Helpers
# -----------------------
def rand_code(k=8):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=k))

def rand_bool():
    return np.random.randint(0, 2)

def rand_conf():
    return np.random.randint(1, 6)

def rand_duration(low=3, high=300):
    return round(np.random.uniform(low, high), 3)

def rand_choice(seq):
    return random.choice(seq)

# -----------------------
# Static values
# -----------------------
DECISION_LABELS = ["good", "poor", "standard"]
GENDERS = ["Female", "Male", "Non-binary / third gender"]
EDUCATION = ["Bachelor’s degree", "Master’s degree", "High school"]
AI_EXP = ["No experience", "Some familiarity", "High familiarity"]


def create_synthetic_data():
    # -----------------------
    # Column construction
    # -----------------------
    columns = [
        "participant.id_in_session",
        "participant.code",
        "participant.label",
        "participant._is_bot",
        "participant._index_in_pages",
        "participant._max_page_index",
        "participant._current_app_name",
        "participant._current_page_name",
        "participant.time_started_utc",
        "participant.visited",
        "participant.mturk_worker_id",
        "participant.mturk_assignment_id",
        "participant.payoff",
        "session.code",
        "session.label",
        "session.mturk_HITId",
        "session.mturk_HITGroupId",
        "session.comment",
        "session.is_demo",
        "session.config.name",
        "session.config.participation_fee",
        "session.config.real_world_currency_per_point",
    ]

    # Consent / instruction / checks
    columns += [
        "consent.1.player.id_in_group",
        "consent.1.player.role",
        "consent.1.player.payoff",
        "consent.1.player.consent_agree",
        "instructions.1.player.id_in_group",
        "instructions.1.player.role",
        "instructions.1.player.payoff",
        "checks.1.player.id_in_group",
        "checks.1.player.role",
        "checks.1.player.payoff",
        "checks.1.player.check_decision_authority",
        "checks.1.player.check_model_understanding",
        "checks.1.player.attempts_decision_authority",
        "checks.1.player.attempts_model_understanding",
        "checks.1.player.failed_checks",
    ]

    # Example trials
    for t in range(1, N_EXAMPLE_TRIALS + 1):
        columns += [
            f"example_trials.{t}.player.id_in_group",
            f"example_trials.{t}.player.role",
            f"example_trials.{t}.player.payoff",
            f"example_trials.{t}.player.page_duration_stage1",
            f"example_trials.{t}.player.page_duration_stage2",
            f"example_trials.{t}.player.case_id",
            f"example_trials.{t}.player.y_true",
            f"example_trials.{t}.player.point_pred_cal",
            f"example_trials.{t}.player.point_pred_confidence",
            f"example_trials.{t}.player.cp_contains_good",
            f"example_trials.{t}.player.cp_contains_poor",
            f"example_trials.{t}.player.cp_contains_standard",
            f"example_trials.{t}.player.initial_decision",
            f"example_trials.{t}.player.initial_confidence",
            f"example_trials.{t}.player.final_decision",
            f"example_trials.{t}.player.final_confidence",
        ]

    # Main trials
    for t in range(1, N_MAIN_TRIALS + 1):
        columns += [
            f"main_trials.{t}.player.id_in_group",
            f"main_trials.{t}.player.role",
            f"main_trials.{t}.player.payoff",
            f"main_trials.{t}.player.page_duration_stage1",
            f"main_trials.{t}.player.page_duration_stage2",
            f"main_trials.{t}.player.case_id",
            f"main_trials.{t}.player.y_true",
            f"main_trials.{t}.player.final_correct",
            f"main_trials.{t}.player.point_pred_cal",
            f"main_trials.{t}.player.point_pred_confidence",
            f"main_trials.{t}.player.cp_contains_good",
            f"main_trials.{t}.player.cp_contains_poor",
            f"main_trials.{t}.player.cp_contains_standard",
            f"main_trials.{t}.player.initial_decision",
            f"main_trials.{t}.player.initial_confidence",
            f"main_trials.{t}.player.final_decision",
            f"main_trials.{t}.player.final_confidence",
        ]

    # Post-experiment measures
    columns += [
        "cognitive_load.1.player.id_in_group",
        "cognitive_load.1.player.role",
        "cognitive_load.1.player.payoff",
        "cognitive_load.1.player.cognitive_load_mental",
        "control_measures.1.player.id_in_group",
        "control_measures.1.player.role",
        "control_measures.1.player.payoff",
        "control_measures.1.player.age",
        "control_measures.1.player.gender",
        "control_measures.1.player.education",
        "control_measures.1.player.ai_literacy_sk9",
        "control_measures.1.player.ai_literacy_sk10",
        "control_measures.1.player.ai_literacy_ail2",
        "control_measures.1.player.ai_literacy_ue2",
        "control_measures.1.player.ai_attitude",
        "control_measures.1.player.ai_trust",
        "control_measures.1.player.risk_aversion",
        "control_measures.1.player.domain_experience",
        "control_measures.1.player.comment",
        "closing.1.player.id_in_group",
        "closing.1.player.role",
        "closing.1.player.payoff",
        "closing.1.player.contact_email",
        "closing.1.player.num_correct_final",
        "consent.1.player.condition",
    ]

    # -----------------------
    # Row generation
    # -----------------------
    rows = []
    start_time = datetime(2026, 2, 4, 12, 0, 0)

    for pid in range(1, N_PARTICIPANTS + 1):
        row = {}

        # Participant/session
        row["participant.id_in_session"] = pid
        row["participant.code"] = rand_code()
        row["participant.label"] = ""
        row["participant._is_bot"] = 0
        row["participant._index_in_pages"] = 66
        row["participant._max_page_index"] = 66
        row["participant._current_app_name"] = "closing"
        row["participant._current_page_name"] = "Closing2"
        row["participant.time_started_utc"] = start_time + timedelta(minutes=pid * 2)
        row["participant.visited"] = 1
        row["participant.mturk_worker_id"] = ""
        row["participant.mturk_assignment_id"] = ""
        row["participant.payoff"] = 0.0

        row["session.code"] = rand_code()
        row["session.label"] = ""
        row["session.mturk_HITId"] = ""
        row["session.mturk_HITGroupId"] = ""
        row["session.comment"] = ""
        row["session.is_demo"] = 0
        row["session.config.name"] = "main_experiment"
        row["session.config.participation_fee"] = 0.0
        row["session.config.real_world_currency_per_point"] = 1.0

        # Consent / checks
        row["consent.1.player.id_in_group"] = pid
        row["consent.1.player.role"] = ""
        row["consent.1.player.payoff"] = 0.0
        row["consent.1.player.consent_agree"] = 1

        row["instructions.1.player.id_in_group"] = pid
        row["instructions.1.player.role"] = ""
        row["instructions.1.player.payoff"] = 0.0

        row["checks.1.player.id_in_group"] = pid
        row["checks.1.player.role"] = ""
        row["checks.1.player.payoff"] = 0.0
        row["checks.1.player.check_decision_authority"] = rand_bool()
        row["checks.1.player.check_model_understanding"] = rand_bool()
        row["checks.1.player.attempts_decision_authority"] = np.random.randint(1, 4)
        row["checks.1.player.attempts_model_understanding"] = np.random.randint(1, 4)
        row["checks.1.player.failed_checks"] = 0

        # Example trials
        for t in range(1, N_EXAMPLE_TRIALS + 1):
            row[f"example_trials.{t}.player.id_in_group"] = pid
            row[f"example_trials.{t}.player.role"] = ""
            row[f"example_trials.{t}.player.payoff"] = 0.0
            row[f"example_trials.{t}.player.page_duration_stage1"] = rand_duration()
            row[f"example_trials.{t}.player.page_duration_stage2"] = rand_duration()
            row[f"example_trials.{t}.player.case_id"] = np.random.randint(1000, 20000)
            row[f"example_trials.{t}.player.y_true"] = rand_choice(DECISION_LABELS)
            row[f"example_trials.{t}.player.point_pred_cal"] = rand_choice(DECISION_LABELS)
            row[f"example_trials.{t}.player.point_pred_confidence"] = rand_conf()
            row[f"example_trials.{t}.player.cp_contains_good"] = rand_bool()
            row[f"example_trials.{t}.player.cp_contains_poor"] = rand_bool()
            row[f"example_trials.{t}.player.cp_contains_standard"] = rand_bool()
            row[f"example_trials.{t}.player.initial_decision"] = np.random.randint(1, 3)
            row[f"example_trials.{t}.player.initial_confidence"] = rand_conf()
            row[f"example_trials.{t}.player.final_decision"] = np.random.randint(1, 3)
            row[f"example_trials.{t}.player.final_confidence"] = rand_conf()

        # Main trials
        num_correct = 0
        for t in range(1, N_MAIN_TRIALS + 1):
            correct = rand_bool()
            num_correct += correct

            row[f"main_trials.{t}.player.id_in_group"] = pid
            row[f"main_trials.{t}.player.role"] = ""
            row[f"main_trials.{t}.player.payoff"] = 0.0
            row[f"main_trials.{t}.player.page_duration_stage1"] = rand_duration()
            row[f"main_trials.{t}.player.page_duration_stage2"] = rand_duration()
            row[f"main_trials.{t}.player.case_id"] = np.random.randint(1000, 20000)
            row[f"main_trials.{t}.player.y_true"] = rand_choice(DECISION_LABELS)
            row[f"main_trials.{t}.player.final_correct"] = rand_bool()
            row[f"main_trials.{t}.player.point_pred_cal"] = rand_choice(DECISION_LABELS)
            row[f"main_trials.{t}.player.point_pred_confidence"] = rand_conf()
            row[f"main_trials.{t}.player.cp_contains_good"] = rand_bool()
            row[f"main_trials.{t}.player.cp_contains_poor"] = rand_bool()
            row[f"main_trials.{t}.player.cp_contains_standard"] = rand_bool()
            row[f"main_trials.{t}.player.initial_decision"] = np.random.randint(1, 3)
            row[f"main_trials.{t}.player.initial_confidence"] = rand_conf()
            row[f"main_trials.{t}.player.final_decision"] = np.random.randint(1, 3)
            row[f"main_trials.{t}.player.final_confidence"] = rand_conf()

        # Post measures
        row["cognitive_load.1.player.id_in_group"] = pid
        row["cognitive_load.1.player.role"] = ""
        row["cognitive_load.1.player.payoff"] = 0.0
        row["cognitive_load.1.player.cognitive_load_mental"] = rand_conf()

        row["control_measures.1.player.id_in_group"] = pid
        row["control_measures.1.player.role"] = ""
        row["control_measures.1.player.payoff"] = 0.0
        row["control_measures.1.player.age"] = np.random.randint(18, 65)
        row["control_measures.1.player.gender"] = rand_choice(GENDERS)
        row["control_measures.1.player.education"] = rand_choice(EDUCATION)
        row["control_measures.1.player.ai_literacy_sk9"] = rand_conf()
        row["control_measures.1.player.ai_literacy_sk10"] = rand_conf()
        row["control_measures.1.player.ai_literacy_ail2"] = rand_conf()
        row["control_measures.1.player.ai_literacy_ue2"] = rand_conf()
        row["control_measures.1.player.ai_attitude"] = rand_conf()
        row["control_measures.1.player.ai_trust"] = rand_conf()
        row["control_measures.1.player.risk_aversion"] = rand_conf()
        row["control_measures.1.player.domain_experience"] = rand_choice(AI_EXP)
        row["control_measures.1.player.comment"] = ""

        row["closing.1.player.id_in_group"] = pid
        row["closing.1.player.role"] = ""
        row["closing.1.player.payoff"] = 0.0
        row["closing.1.player.contact_email"] = f"user{pid}@example.com"
        row["closing.1.player.num_correct_final"] = num_correct
        row["consent.1.player.condition"] = np.random.randint(1, 4)

        rows.append(row)

    # -----------------------
    # Create & export CSV
    # -----------------------
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(RAW_DATA_DIR / "all_apps_wide-synthetic_experiment_data.csv", index=False)

    print("CSV written: synthetic_experiment_data.csv")


if __name__ == '__main__':
    create_synthetic_data()
