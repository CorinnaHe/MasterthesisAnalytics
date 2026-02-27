import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


def plot_binary_col_by_ordinal_col(df: pd.DataFrame, binary_col_name: str, ordinal_col_name: str):
    # Convert boolean to 0/1
    df["binary_bin"] = df[binary_col_name].astype(int)

    # Fit logistic model
    model = smf.glm(
        formula=f"binary_bin ~ {ordinal_col_name}",
        data=df,
        family=sm.families.Binomial()
    ).fit()

    print(model.summary())

    observed = (
        df.groupby(ordinal_col_name)["binary_bin"]
        .mean()
        .reset_index()
    )

    min_val = df[ordinal_col_name].min()
    max_val = df[ordinal_col_name].max()

    x_pred = np.linspace(min_val, max_val, 100)
    pred_df = pd.DataFrame({ordinal_col_name: x_pred})
    pred_df["predicted"] = model.predict(pred_df)

    plt.figure()

    # Observed points
    plt.scatter(
        observed[ordinal_col_name],
        observed["binary_bin"]
    )

    # Logistic curve
    plt.plot(
        x_pred,
        pred_df["predicted"]
    )

    plt.xlabel(ordinal_col_name)
    plt.ylabel(binary_col_name)
    plt.ylim(0, 1)
    plt.xticks( list(range(int(min_val), int(max_val) + 1)))
    plt.title(f"{binary_col_name} as a Function of {ordinal_col_name}")

    plt.show()