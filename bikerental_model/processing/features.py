from typing import List
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class WeekdayImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weekday' column by extracting dayname from 'dteday' column """

    def __init__(self, variables: str, date_column: str = 'dteday'):
        if not isinstance(variables, str):
            raise ValueError("variables should be a string")
        self.variables = variables
        self.date_column = date_column
        self.fill_value = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        X = X.copy()
        X[self.date_column] = pd.to_datetime(X[self.date_column], errors='coerce')
        # self.fill_value = X[self.variables].mode()[0]  # Store the most frequent weekday
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.date_column] = pd.to_datetime(X[self.date_column], errors='coerce')

        # Find missing indices
        missing_idx = X[X[self.variables].isnull()].index
        
        # Impute missing values using day names
        X.loc[missing_idx, self.variables] = (
            X.loc[missing_idx, self.date_column].dt.day_name().str[:3]
        )
        
        # Fill remaining missing values with the most frequent weekday
        # X[self.variables] = X[self.variables].fillna(self.fill_value)
        
        return X



class WeathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

    def __init__(self, variables: str):
        # YOUR CODE HERE
        if not isinstance(variables, str):
            raise ValueError("variables should be a string")

        self.variables = variables
        self.fill_value = "Mist"

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        # self.fill_value=X[self.variables].mode()[0]
        mode_value = X[self.variables].mode()
        self.fill_value = mode_value[0] if not mode_value.empty else "Mist"
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # YOUR CODE HERE
        X = X.copy()
        X[self.variables]=X[self.variables].fillna(self.fill_value)

        return X



class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper:
    Treat column as Ordinal categorical variable, and assign values accordingly
    """

    def __init__(self, variables: str, mappings: dict):

        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        self.variables = variables
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        #for feature in self.variables:
        X[self.variables] = X[self.variables].map(self.mappings).astype(int)

        return X



class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """   

    def __init__(self, variables=['temp', 'atemp', 'hum', 'windspeed'], method="iqr", factor=1.5):
        """
        Parameters:
        - variables: List of numerical columns to handle outliers.
        - method: Outlier detection method ('iqr' or 'zscore').
        - factor: Threshold factor for detecting outliers (default 1.5 for IQR).
        """
        
        self.variables = variables
        self.method = method
        self.factor = factor
        self.bounds = {}  # Store lower and upper bounds

    def fit(self, X: pd.DataFrame, y=None):
        """Learn the lower and upper bounds for outlier detection."""
        X = X.copy()
        if self.variables is None:
            self.variables = X.select_dtypes(include=[np.number]).columns  # Use all numerical columns

        for var in self.variables:
            if self.method == "iqr":
                Q1 = X[var].quantile(0.25)
                Q3 = X[var].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.factor * IQR
                upper_bound = Q3 + self.factor * IQR
            elif self.method == "zscore":
                mean = X[var].mean()
                std = X[var].std()
                lower_bound = mean - self.factor * std
                upper_bound = mean + self.factor * std
            else:
                raise ValueError("Method must be 'iqr' or 'zscore'.")

            self.bounds[var] = (lower_bound, upper_bound)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply outlier handling by capping values within bounds."""
        X = X.copy()
        #print(X)
        #self = self.fit(X)
        #print(self.bounds)
        
        for var in self.variables:
            #print(var)
            lower_bound, upper_bound = self.bounds[var]
            X[var] = np.where(X[var] < lower_bound, lower_bound, X[var])
            X[var] = np.where(X[var] > upper_bound, upper_bound, X[var])
        return X



class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """One-hot encode the 'weekday' column."""

    def __init__(self, variables="weekday", drop_first=False):
        """
        Parameters:
        - variables: Column name to encode (default 'weekday').
        - drop_first: Whether to drop the first category to avoid multicollinearity.
        """
        self.variables = variables
        self.drop_first = drop_first
        self.categories_ = None  # Store unique values

    def fit(self, X: pd.DataFrame, y=None):
        """Learn the unique categories for one-hot encoding."""
        self.categories_ = sorted(X[self.variables].dropna().unique())  # Store unique weekday values
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply one-hot encoding to the 'weekday' column."""
        X = X.copy()
        X = pd.get_dummies(X, columns=[self.variables], drop_first=self.drop_first)
        return X



class NumericColumnSelector(BaseEstimator, TransformerMixin):
    """Custom transformer to keep only numerical columns (int or float)."""
    
    def fit(self, X, y=None):
        """Fit method stores the names of numeric columns for transformation."""
        self.numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
        return self

    def transform(self, X):
        """Drops non-numeric columns and returns only numeric features."""
        #numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
        return X[self.numeric_columns].copy()