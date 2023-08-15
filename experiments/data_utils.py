import pandas as pd
import os, sys
import numpy as np


# Sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, TransformerMixin

sys.path.append(os.path.abspath(".."))
from src.features import Features


class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, df):
        """ I assume that the last column is the label """
        ncols = df.shape[1] - 1
        target = df.columns[-1]
        self.categories_ = [0] * ncols
        for i, feature in enumerate(df.columns[:-1]):
            frequencies = []
            self.categories_[i] = np.unique(df[feature])
            for category in self.categories_[i]:
                frequencies.append(df[df[feature]==category][target].mean())
            argsort = np.argsort(frequencies)
            self.categories_[i] = self.categories_[i][argsort]
        return self


    def transform(self, df):
        """ I assume that the last column is the label """
        ncols = df.shape[1] - 1
        X = np.zeros((df.shape[0], ncols))
        for i, feature in enumerate(df.columns[:-1]):
            for j, category in enumerate(self.categories_[i]):
                idx = np.where(df[feature]==category)[0]
                X[idx, i] = j
        return X



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




def get_data_adults():

    # load train
    raw_data_1 = np.genfromtxt(os.path.join(os.path.dirname(__file__), 'datasets', 
                                            'Adult-Income','adult.data'), 
                                                     delimiter=', ', dtype=str)
    # load test
    raw_data_2 = np.genfromtxt(os.path.join(os.path.dirname(__file__), 'datasets', 
                                            'Adult-Income','adult.test'),
                                      delimiter=', ', dtype=str, skip_header=1)

    feature_names = ['age', 'workclass', 'fnlwgt', 'education',
                     'educational-num', 'marital-status', 'occupation', 
                     'relationship', 'race', 'gender', 'capital-gain', 
                     'capital-loss', 'hours-per-week', 'native-country', 'income']

    # Shuffle train/test
    df = pd.DataFrame(np.vstack((raw_data_1, raw_data_2)), columns=feature_names)


    # For more details on how the below transformations 
    df = df.astype({"age": np.int64, "educational-num": np.int64, 
                    "hours-per-week": np.int64, "capital-gain": np.int64, 
                    "capital-loss": np.int64 })

    # Reduce number of categories
    df = df.replace({'workclass': {'Without-pay': 'Other/Unknown', 
                                   'Never-worked': 'Other/Unknown'}})
    df = df.replace({'workclass': {'?': 'Other/Unknown'}})
    df = df.replace({'workclass': {'Federal-gov': 'Government', 
                                   'State-gov': 'Government', 'Local-gov':'Government'}})
    df = df.replace({'workclass': {'Self-emp-not-inc': 'Self-Employed', 
                                   'Self-emp-inc': 'Self-Employed'}})

    df = df.replace({'occupation': {'Adm-clerical': 'White-Collar', 
                                    'Craft-repair': 'Blue-Collar',
                                    'Exec-managerial':'White-Collar',
                                    'Farming-fishing':'Blue-Collar',
                                    'Handlers-cleaners':'Blue-Collar',
                                    'Machine-op-inspct':'Blue-Collar',
                                    'Other-service':'Service',
                                    'Priv-house-serv':'Service',
                                    'Prof-specialty':'Professional',
                                    'Protective-serv':'Service',
                                    'Tech-support':'Service',
                                    'Transport-moving':'Blue-Collar',
                                    'Unknown':'Other/Unknown',
                                    'Armed-Forces':'Other/Unknown',
                                    '?':'Other/Unknown'}})

    df = df.replace({'marital-status': {'Married-civ-spouse': 'Married', 
                                        'Married-AF-spouse': 'Married', 
                                        'Married-spouse-absent':'Married',
                                        'Never-married':'Single'}})

    df = df.replace({'income': {'<=50K': 0, '<=50K.': 0,  '>50K': 1, '>50K.': 1}})

    df = df.replace({'education': {'Assoc-voc': 'Assoc', 'Assoc-acdm': 'Assoc',
                                   '11th':'School', '10th':'School', 
                                   '7th-8th':'School', '9th':'School',
                                   '12th':'School', '5th-6th':'School', 
                                   '1st-4th':'School', 'Preschool':'School'}})

    # Put numeric+ordinal before nominal and remove fnlwgt-country
    df = df[['age', 'educational-num', 'capital-gain', 'capital-loss',
             'hours-per-week', 'gender', 'workclass','education', 'marital-status', 
             'occupation', 'relationship', 'race', 'income']]
    df = shuffle(df, random_state=42)
    feature_names = df.columns[:-1]

    # Make a column transformer for ordinal encoder
    encoder = ColumnTransformer(transformers=
                      [('identity', FunctionTransformer(), df.columns[:5]),
                       ('encoder', TargetEncoder(), df.columns[5:])
                      ])
    X = encoder.fit_transform(df)
    y = df["income"].to_numpy()
    
    # Generate Features object
    feature_types = ["num", "num", "sparse_num", "sparse_num", "num"] +\
        [(["ordinal"] + list(l)) for l in encoder.transformers_[1][1].categories_]
    features = Features(X, feature_names, feature_types)
    
    return X, y, features




def get_data_california_housing():
    data = fetch_california_housing()

    X, y, feature_names = data["data"], data["target"], data["feature_names"]

    # Remove outlier
    keep_bool = X[:, 5] < 1000
    X = X[keep_bool]
    y = y[keep_bool]
    del keep_bool

    # Take log of right-skewed features
    for i in [2, 3, 5]:
        X[:, i] = np.log10(X[:, i])
        feature_names[i] = f"log_{feature_names[i]}"

    # Add additionnal location feature
    def closest_point(location):
        # Biggest cities in 1990
        # Los Angeles, San Francisco, San Diego, San Jose
        biggest_cities = [
            (34.052235, -118.243683),
            (37.773972, -122.431297),
            (32.715736, -117.161087),
            (37.352390, -121.953079),
        ]
        closest_location = None
        for city_x, city_y in biggest_cities:
            distance = ((city_x - location[0]) ** 2 + (city_y - location[1]) ** 2) ** (
                1 / 2
            )
            if closest_location is None:
                closest_location = distance
            elif distance < closest_location:
                closest_location = distance
        return closest_location

    X = np.column_stack((X, [closest_point(x[-2:]) for x in X]))
    feature_names.append('ClosestBigCityDist')

    # Generate Features object
    feature_types = ["num", "num", "num", "num",\
                     "num", "num", "num", "num", "num"]
    
    features = Features(X, feature_names, feature_types)
    
    return X, y, features




DATASET_MAPPING = {
    "bike": get_data_bike,
    "california": get_data_california_housing,
    "adult_income" : get_data_adults,
}

TASK_MAPPING = {
    "bike": "regression",
    "california" : "regression",
    "adult_income": "classification",
}

INTERACTIONS_MAPPING = {
    "bike" : [0, 2, 5, 7],
    "adult_income" : [0, 4, 5, 8, 10]
}

SCATTER_SHOW = {
    "bike" : [2, 7],
    "adult_income": [0, 1, 4, 5]
}