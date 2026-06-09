import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import joblib
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


def train_xgboost(
    X_train,
    y_train,
    X_val,
    y_val
):

    with mlflow.start_run(
        run_name="xgboost"
    ):

        params = {
            "learning_rate": 0.05,
            "max_depth": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_estimators": 2000,
            "random_state": 42
        }

        mlflow.log_params(params)

        model = xgb.XGBRegressor(
            **params
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[
                (X_val, y_val)
            ],
            verbose=False
        )

        preds = model.predict(
            X_val
        )

        mae = mean_absolute_error(
            y_val,
            preds
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_val,
                preds
            )
        )

        mlflow.log_metric(
            "MAE",
            mae
        )

        mlflow.log_metric(
            "RMSE",
            rmse
        )

        mlflow.sklearn.log_model(
            model,
            "model"
        )

        joblib.dump(
            {
                "model": model,
                "features": list(
                    X_train.columns
                )
            },
            "models/xgboost.pkl"
        )

        print(
            f"XGBoost MAE: {mae:.4f}"
        )

        print(
            f"XGBoost RMSE: {rmse:.4f}"
        )

        return model


def train_lightgbm(
    X_train,
    y_train,
    X_val,
    y_val
):

    with mlflow.start_run(
        run_name="lightgbm"
    ):

        params = {
            "learning_rate": 0.05,
            "num_leaves": 128,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "n_estimators": 2000,
            "random_state": 42
        }

        mlflow.log_params(
            params
        )

        model = lgb.LGBMRegressor(
            **params
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[
                (X_val, y_val)
            ],
            callbacks=[
                lgb.early_stopping(100)
            ]
        )

        preds = model.predict(
            X_val
        )

        mae = mean_absolute_error(
            y_val,
            preds
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_val,
                preds
            )
        )

        mlflow.log_metric(
            "MAE",
            mae
        )

        mlflow.log_metric(
            "RMSE",
            rmse
        )

        mlflow.sklearn.log_model(
            model,
            "model"
        )

        joblib.dump(
            {
                "model": model,
                "features": list(
                    X_train.columns
                )
            },
            "models/lightgbm.pkl"
        )

        print(
            f"LightGBM MAE: {mae:.4f}"
        )

        print(
            f"LightGBM RMSE: {rmse:.4f}"
        )

        return model