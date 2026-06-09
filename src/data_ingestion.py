import pandas as pd
import os

def load_data(data_dir="data"):
    
    calendar = pd.read_csv(f"{data_dir}/calendar.csv")
    sales = pd.read_csv(
    f"{data_dir}/sales_train_validation.csv")


    sales = sales.sample(
    n=2000,
    random_state=42)

    prices = pd.read_csv(f"{data_dir}/sell_prices.csv")

    id_vars = [
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id"
    ]

    sales_long = pd.melt(
        sales,
        id_vars=id_vars,
        var_name="d",
        value_name="sales"
    )

    df = sales_long.merge(calendar, on="d", how="left")

    df = df.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left"
    )

    df["date"] = pd.to_datetime(df["date"])

    return df.sort_values(["id", "date"])