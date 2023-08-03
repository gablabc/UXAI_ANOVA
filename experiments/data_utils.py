import pandas as pd
import os, sys

sys.path.append(os.path.abspath(".."))
from src.features import Features


def get_data_bike():
    df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "datasets", "Bike-Sharing/hour.csv")
    )
    df.drop(columns=["dteday", "casual", "registered", "instant"], inplace=True)

    # Remove correlated features
    df.drop(columns=["atemp", "season"], inplace=True)

    # Rescale temp to Celcius
    df["temp"] = 41 * df["temp"]

    # Month count starts at 0
    df["mnth"] -= 1

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42)

    # Scale all features
    feature_names = list(df.columns[:-1])

    X = df.to_numpy()[:, :-1]
    y = df.to_numpy()[:, -1]

    # Generate Features object
    feature_types = [
        ["ordinal", "2011", "2012"],
        ["ordinal",
            "January", "February", "March", "April", "May", "June", "July",
            "August", "September", "October","November", "December",
        ],
        "num_int",
        "bool",
        ["ordinal",
            "Sunday", "Monday", "Thuesday", "Wednesday", "Thursday",
            "Friday", "Saturday"],
        "bool",
        "num_int",
        "num",
        "num",
        "num",
    ]

    features = Features(X, feature_names, feature_types)

    return X, y, features


DATASET_MAPPING = {
    "bike": get_data_bike,
}

TASK_MAPPING = {
    "bike": "regression",
}
