import pandas as pd
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def test_h2(df: pd.DataFrame):
    vcf = {
        "participant": "0 + C(participant_code)"
    }



    model_h2 = BinomialBayesMixedGLM.from_formula(
        "appropriate_reliance ~ is_set_based",
        vcf,
        df
    )

    result_h2 = model_h2.fit_vb()
    print(result_h2.summary())


def test_initial_ai_agree_and_switching_regulate_confidence(df: pd.DataFrame):
    df["decision_label"] = df["decision_label"].astype("category")
    df["condition"] = df["condition"].astype("category")
    df["participant_code"] = df["participant_code"].astype("category")

    model = smf.mixedlm(
        "delta_confidence ~ decision_label * condition",
        data=df,
        groups=df["participant_code"],
    )

    result = model.fit(method="lbfgs")

    print(result.summary())

    tukey = pairwise_tukeyhsd(
        endog=df["delta_confidence"],
        groups=df["decision_label"],
        alpha=0.05
    )

    print(tukey)


