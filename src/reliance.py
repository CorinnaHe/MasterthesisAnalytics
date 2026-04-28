import pandas as pd
import pingouin as pg
import itertools
from scipy.stats import chi2_contingency
import sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


from data_loader import load_experiment_data
from figures import plot_box_with_jitter, plot_binary_stacked_bar
from variable_constructer import construct_trial_level_variables


def pairwise_chi2_tests(table: pd.DataFrame, alpha=0.1):
    """
    Performs pairwise Chi-square tests between rows of a contingency table
    with Bonferroni correction.

    Parameters:
        table (pd.DataFrame): contingency table (rows = groups, columns = outcomes)
        alpha (float): significance level

    Returns:
        pd.DataFrame with results
    """

    conditions = table.index.tolist()
    pairs = list(itertools.combinations(conditions, 2))

    results = []
    m = len(pairs)  # number of comparisons
    alpha_corrected = alpha / m

    for c1, c2 in pairs:
        subtable = table.loc[[c1, c2]]

        chi2, p, dof, expected = chi2_contingency(subtable)

        results.append({
            "Comparison": f"{c1} vs {c2}",
            "Chi2": chi2,
            "p-value": p,
            "p-value (Bonferroni)": min(p * m, 1.0),
            "Significant (uncorrected)": p < alpha,
            "Significant (Bonferroni)": p < alpha_corrected
        })

    return pd.DataFrame(results), alpha_corrected


def run_mixed_anova(df):
    aov = pg.mixed_anova(
        dv="delta_confidence",
        within="decision_label",
        between="condition",
        subject="participant_code",
        data=df
    )

    print(aov)
    return aov

def run_tukey_tests(df):
    posthoc = pg.pairwise_tukey(
        data=df,
        dv="delta_confidence",
        between="decision_label"
    )

    print(posthoc)
    return posthoc

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

    # ------------------------------------------------------------------
    # 6. Statistical Analysis like 4.2.2
    # ------------------------------------------------------------------

    run_mixed_anova(trials)
    run_tukey_tests(trials)

    # replace ANOVA with linear mixed model to support unbalanced category data
    model = smf.mixedlm(
        "delta_confidence ~ decision_label * condition",
        trials,
        groups=trials["participant_code"]
    )

    result = model.fit()
    print(result.summary())

def inspect_reliance_based_on_condition(mismatch_df: pd.DataFrame):
    conditions = mismatch_df["condition"].unique()

    results = {}

    for cond in conditions:
        data = mismatch_df[mismatch_df["condition"] == cond]

        # logistic regression
        model = smf.logit(
            #final_agree_ai here is similar to switch_to_ai bc mismatch df
            "final_agree_ai ~ shared_ai_confidence", #Cao et al. uses raw conf, we use unified measure
            data=data
        ).fit(
            # added to Cao et al. clustered standard errors by participant
            cov_type="cluster",
            cov_kwds={"groups": data["participant_code"]}
        )

        results[cond] = model

        print("\n============================")
        print("Condition:", cond)
        print("============================")

        print(model.summary())

        # Likelihood ratio test (Chi-square)
        lr_stat = model.llr
        p_val = model.llr_pvalue
        df_model = model.df_model

        print(f"\nLikelihood ratio χ²({int(df_model)}) = {lr_stat:.2f}")
        print(f"p-value = {p_val:.3f}")

def inspect_appropriate_reliance_based_on_condition(df: pd.DataFrame, reliance_colum: str):
    print(df.groupby("condition")[reliance_colum].mean())

    model = smf.logit(
        f"{reliance_colum} ~ C(condition) + C(case_id)",
        data=df
    )
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": df["participant_code"]},
        disp=False
    )
    print(result.summary())

    table = pd.crosstab(
        df["condition"],
        df[reliance_colum]
    )
    print(table)

    chi2, p, dof, expected = stats.chi2_contingency(table)

    print("Chi-square:", chi2)
    print("p-value:", p)
    print("df:", dof)

    print("\n=== Pairwise Chi-square (Bonferroni corrected) ===")
    pairwise_results, alpha_corr = pairwise_chi2_tests(table)

    print(f"Corrected alpha: {alpha_corr:.4f}")
    print(pairwise_results)

    # additional to Cao et al.
    #print("\n=== By Strategy ===")
    #print(df.groupby(["condition", "strategy"])[reliance_colum].mean())

    model = smf.logit(
        f"{reliance_colum} ~ C(condition) * C(strategy)",
        data=df
    )
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": df["participant_code"]},
        disp=False
    )
    #print(result.summary())

    model = smf.logit(
        f"{reliance_colum} ~ confidence_gap * C(condition)",
        data=df
    )
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": df["participant_code"]},
        disp=False
    )
    #print(result.summary())


def inspect_confidence_change_based_on_condition(cases):
    for name, subset in cases.items():
        model = smf.ols(
            "delta_confidence ~ C(condition)",
            data=subset
        ).fit()

        anova = sm.stats.anova_lm(model, typ=2)

        print("\nCASE:", name)
        print(anova)

        # run Tukey if ANOVA significant
        p_value = anova.loc["C(condition)", "PR(>F)"]

        if p_value < 0.05:
            print("\nTukey HSD post-hoc test:")

            tukey = pairwise_tukeyhsd(
                endog=subset["delta_confidence"],
                groups=subset["condition"],
                alpha=0.05
            )

            print(tukey.summary())

        # One-way ANOVA ignores clustering so we additionally to Cao et al. do a mixed effects model:
        model = smf.mixedlm(
            "delta_confidence ~ C(condition)",
            data=subset,
            groups=subset["participant_code"]
        )

        result = model.fit()

        print(result.summary())


import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def inspect_switching_behaviour(df):

    print("\n======================================")
    print("SECTION 5 – SWITCHING BEHAVIOUR MODEL")
    print("======================================")

    df = df[df["initial_agree_ai"] == 0].copy()

    predictors = [
        "initial_decision",
        "initial_confidence",
        "shared_ai_confidence",
        "C(condition)",
        "age",
        "C(gender)",
        "C(education)",
        "ai_literacy_sk9",
        "ai_literacy_sk10",
        "ai_literacy_ail2",
        "ai_literacy_ue2",
        "C(domain_experience)",
        "risk_aversion"
    ]

    def build_formula(preds):
        return "final_agree_ai ~ " + " + ".join(preds)

    def fit_model(formula):
        # original Cao et al. return smf.logit(formula, data=df).fit(disp=False)

        return smf.logit(formula, data=df).fit(
            cov_type="cluster",
            cov_kwds={"groups": df["participant_code"]},
            disp=False
        )

    current_predictors = predictors.copy()
    formula = build_formula(current_predictors)
    current_model = fit_model(formula)

    print("\nFULL MODEL")
    print(current_model.summary())

    print("\n--- BACKWARD ELIMINATION ---")

    while True:

        pvalues = current_model.pvalues.drop("Intercept")
        worst_term = pvalues.idxmax()
        worst_p = pvalues.max()

        if worst_p < 0.25:
            break

        # map dummy terms back to original predictor
        base_var = worst_term.split("[")[0]

        if base_var not in current_predictors:
            # safeguard to prevent infinite loops
            print(f"Stopping: could not map {worst_term}")
            break

        print(f"\nRemoving predictor: {base_var} (p={worst_p:.3f})")

        reduced_predictors = [v for v in current_predictors if v != base_var]

        reduced_formula = build_formula(reduced_predictors)
        reduced_model = fit_model(reduced_formula)

        lr = 2 * (current_model.llf - reduced_model.llf)
        df_diff = current_model.df_model - reduced_model.df_model
        p_lr = stats.chi2.sf(lr, df_diff)

        print(f"LR test: χ² = {lr:.3f}, p = {p_lr:.3f}")

        current_predictors = reduced_predictors
        current_model = reduced_model

    print("\n======================================")
    print("FINAL MODEL")
    print("======================================")

    print(current_model.summary())
    print("\nMcFadden pseudo R²:", round(current_model.prsquared, 3))

    return current_model


if __name__ == '__main__':
    experiment_date = "2026-03-20"
    (
        main_trials_df,
        control_measures_df,
        participant_stats,
        *_

    ) = load_experiment_data(f"all_apps_wide-{experiment_date}.csv")
    main_trials_df = main_trials_df.merge(participant_stats[['participant_code', 'strategy']], on='participant_code', how='left')
    main_trials_df["final_agree_ai"] = main_trials_df["final_agree_ai"].astype(int)

    # Cao et al. 4.2
    #inspect_human_ai_match(main_trials_df, "general")

    mismatch_df = main_trials_df[
        main_trials_df["initial_agree_ai"] == 0
        ].copy()

    # Cao et al. 4.3.1
    #inspect_reliance_based_on_condition(mismatch_df)

    # Cao et al. 4.3.2 -> Overreliance
    print("=== Overreliance ===")
    ai_wrong_df = mismatch_df[mismatch_df["ai_correct"] == False]
    inspect_appropriate_reliance_based_on_condition(ai_wrong_df, "final_agree_ai")

    print("=== Underreliance ===")
    # added to Cao et al. -> Underreliance
    ai_correct_df = mismatch_df[mismatch_df["ai_correct"] == True]
    inspect_appropriate_reliance_based_on_condition(ai_correct_df, "final_agree_ai")

    print("=== Appropriate Reliance ===")
    # addded to Cao et al. -> Appropriate Reliance
    inspect_appropriate_reliance_based_on_condition(mismatch_df, "appropriate_reliance")

    #added
    print("=== AI Confidence vs. AI Correctness ===")
    model = smf.logit(
        f"final_agree_ai ~ ai_correct * shared_ai_confidence",
        data=mismatch_df,
    )
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": mismatch_df["participant_code"]},
        disp=False
    )
    print(result.summary())

    # added
    print("=== Initial Confidence vs. AI Correctness ===")
    model = smf.logit(
        f"final_agree_ai ~ ai_correct * initial_confidence",
        data=mismatch_df,
    )
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": mismatch_df["participant_code"]},
        disp=False
    )
    print(result.summary())

    print("=== AI Confidence vs. AI Correctness ===")
    model = smf.logit(
        f"final_agree_ai ~ ai_correct * shared_ai_confidence",
        data=mismatch_df,
    )
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": mismatch_df["participant_code"]},
        disp=False
    )
    print(result.summary())

    # added
    c3_mismatch = mismatch_df[
        mismatch_df["condition"] == "C3"
    ].copy()
    print("=== Initial pos in set vs. AI Correctness ===")
    model = smf.logit(
        f"final_agree_ai ~ ai_correct * initial_pos_in_set",
        data=c3_mismatch,
    )
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": c3_mismatch["participant_code"]},
        disp=False
    )
    print(result.summary())

    # Cao et al. 4.3.3
    cases = {
        "match_correct":
            main_trials_df[(main_trials_df["initial_agree_ai"] == True) & (main_trials_df["final_correct"] == 1)],

        "match_incorrect":
            main_trials_df[(main_trials_df["initial_agree_ai"] == True) & (main_trials_df["final_correct"] == 0)],

        "mismatch_correct":
            main_trials_df[(main_trials_df["initial_agree_ai"] == False) & (main_trials_df["final_correct"] == 1)],

        "mismatch_incorrect":
            main_trials_df[(main_trials_df["initial_agree_ai"] == False) & (main_trials_df["final_correct"] == 0)]
    }
    inspect_confidence_change_based_on_condition(cases)

    # Cao et al. 4.3.4 -> not applicable here because metrics weren't collected

    # Cao et al. 5
    main_trials_control_measures_df = main_trials_df.merge(control_measures_df, on='participant_code', how='left') # join control measures
    inspect_switching_behaviour(main_trials_control_measures_df)

