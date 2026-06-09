import shap
import pandas as pd

def get_shap_explanations(
    model,
    X
):

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = (
        explainer.shap_values(X)
    )

    return pd.DataFrame({
        "Feature": X.columns,
        "SHAP Value":
        shap_values[0]
    })