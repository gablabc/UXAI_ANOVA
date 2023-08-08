import numpy as np
from copy import deepcopy

# Mappers take feature values and assing them a high level representation
# e.g. numerical -> (low, medium, high), categorical 3 -> "Married" etc.
# They are primarily used when visualising local explanations, where the 
# feature values used by the model may not be easily interpretable.


# Boolean feature
class bool_value_mapper(object):
    """ Organise feature values as 1->true or 0->false """

    def __init__(self):
        self.values = ["False", "True"]

    # map 0->False  1->True
    def __call__(self, x):
        return self.values[round(x)]


# Ordinal/Nominal encoding of categorical features
class cat_value_mapper(object):
    """ Organise categorical features  int_value->'string_value' """

    def __init__(self, categories_in_order):
        self.cats = categories_in_order

    # x takes values 0, 1, 2 ,3  return the category
    def __call__(self, x):
        return self.cats[round(x)]


# Numerical features x in [xmin, xmax]
class numerical_value_mapper(object):
    """ Organise feature values in quantiles  value->{low, medium, high}"""

    def __init__(self, num_feature_values):
        self.quantiles = np.quantile(num_feature_values, [0, 0.2, 0.4, 0.6, 0.8, 1])
        self.quantifiers = ["very small", "small", "medium", "large", "very large"]

    # map feature value to very small, small, medium, large, very large
    def __call__(self, x):
        return self.quantifiers[
            np.where((x <= self.quantiles[1:]) & (x >= self.quantiles[0:-1]))[0][0]
        ]


# Numerical features but with lots of zeros x in {0} U [xmin, xmax]
class sparse_numerical_value_mapper(object):
    """ Organise feature values in quantiles but treat 0-values differently
    """

    def __init__(self, num_feature_values):
        idx = np.where(num_feature_values != 0)[0]
        self.quantiles = np.quantile(
            num_feature_values[idx], [0, 0.2, 0.4, 0.6, 0.8, 1]
        )
        self.quantifiers = ["very small", "small", "medium", "large", "very large"]

    # map feature value to very small, small, medium, large, very large
    def __call__(self, x):
        if x == 0:
            return int(x)
        else:
            return self.quantifiers[
                np.where((x <= self.quantiles[1:]) & (x >= self.quantiles[0:-1]))[0][0]
            ]


# Numerical features with integer values
class numerical_integer_mapper(object):
    """ Simply round the feature into a integer """

    def __init__(self):
        pass

    # map feature 1.0000 to 1
    def __call__(self, x):
        return round(x)


class Features(object):
    """ Abstraction of the concept of a feature. Useful when doing feature importance plots """

    def __init__(self, X, feature_names, feature_types):
        self.names = feature_names
        self.types = []
        # Nominal categorical features that will need to be encoded
        self.nominal = []
        # map feature values to interpretable text
        self.maps = []
        for i, feature_type in enumerate(feature_types):
            # If its a list then the feature is categorical
            if type(feature_type) == list:
                self.types.append(feature_type[0]) # ordinal or nominal
                self.maps.append(cat_value_mapper(feature_type[1:]))
                if feature_type[0] == "nominal":
                    self.nominal.append(i)
            else:   
                self.types.append(feature_type)
                if feature_type == "num":
                    self.maps.append(numerical_value_mapper(X[:, i]))
                    
                elif feature_type == "sparse_num":
                    self.maps.append(sparse_numerical_value_mapper(X[:, i]))
                    
                elif feature_type == "bool":
                    self.maps.append(bool_value_mapper())
                    
                elif feature_type == "num_int":
                    self.maps.append(numerical_integer_mapper())
                else:
                    raise ValueError("Wrong feature type")
                    
        # non_nominal refer to numerical+ordinal features i.e. all features that
        # are naturally represented with numbers
        self.non_nominal = list( set(range(len(feature_types))) - 
                                 set(self.nominal) )
    
    def map_values(self, x):
        """ Map values of x into interpretable text """
        return [f"{self.names[i]}={self.maps[i](x[i])}" for i in range(len(x))]

    def __len__(self):
        return len(self.names)
    

    def select(self, i_range):
        feature_copy = deepcopy(self)
        feature_copy.names = [feature_copy.names[i] for i in i_range]
        feature_copy.types = [feature_copy.types[i] for i in i_range]
        feature_copy.maps = [feature_copy.maps[i] for i in i_range]
        # TODO handle nominal and non_nominal
        feature_copy.nominal = []
        feature_copy.non_nominal = []
        return feature_copy