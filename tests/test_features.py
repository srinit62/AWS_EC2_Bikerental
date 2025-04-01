
"""
Note: These tests will fail if you have not first trained the model.

Implement test cases for:
    ● Pipeline processing steps, including imputation, mapping, and custom class
        transformations
    ● Prediction steps

"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from bikerental_model.config.core import config
from bikerental_model.processing.features import WeekdayImputer, WeathersitImputer, NumericColumnSelector, OutlierHandler


def test_weekday_variable_transformer(sample_input_data):
    # Given
    transformer = WeekdayImputer(
        variables=config.model_config_.features[9],  # cabin
    )
    assert np.isnan(sample_input_data[0].loc[7046,'weekday'])

    # When
    subject = transformer.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[7046, 'weekday'] == 'Wed'

def test_weathersit_variable_transformer(sample_input_data):
    # Given
    transformer = WeathersitImputer(
        variables=config.model_config_.features[11],  # cabin
    )
    assert np.isnan(sample_input_data[0].loc[7046,'weathersit'])

    # When
    subject = transformer.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[7046, 'weathersit'] == 'Clear'


#Numeric column validator

def test_numeric_column_transformer(sample_input_data):
    #transformer = NumericColumnSelector()
    #numeric_cols = transformer.fit(sample_input_data[2])
    #numeric_columns = list(numeric_cols.numeric_columns)
    #print(numeric_columns)
    
    #Hardcoded original list of numeric features as more features are 
    # converted to numeric in the pipeline
    numeric_columns = ['temp', 'atemp', 'hum', 'windspeed']

    #Get boundsfrom Train Dataset
    #--------------------------------------------------
    transformer = OutlierHandler(numeric_columns)
    data_bounds = transformer.fit(sample_input_data[2])
    num_col_bounds = data_bounds.bounds    

    #Fit and Transform Test Dataset
    #--------------------------------------------------       
    for var in numeric_columns:
        var_list = []
        var_list.append(var)

        lower_bound, upper_bound = num_col_bounds[var]

        list_of_test_outliers = list(sample_input_data[0].loc[(sample_input_data[0][var] < lower_bound) | (sample_input_data[0][var] > upper_bound)].index)        
        #print(var)
        #print(list_of_test_outliers)
        #print('-----------------------------------')

        #Given
        transformer = OutlierHandler(variables=var_list)
        #assert len(list_of_test_outliers) > 0
        if len(list_of_test_outliers) > 0:
            #When
            subject = transformer.fit(sample_input_data[0]).transform(sample_input_data[0])

            #Then
            validate_outliers = list(subject.loc[(subject[var] < lower_bound) | (subject[var] > upper_bound)].index)
            assert len(validate_outliers) == 0
    

