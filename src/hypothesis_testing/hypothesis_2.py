import pandas as pd
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

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


