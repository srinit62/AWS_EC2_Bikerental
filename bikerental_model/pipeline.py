import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from bikerental_model.config.core import config
from bikerental_model.processing.features import WeekdayImputer
from bikerental_model.processing.features import WeathersitImputer
from bikerental_model.processing.features import Mapper
from bikerental_model.processing.features import OutlierHandler
from bikerental_model.processing.features import WeekdayOneHotEncoder
from bikerental_model.processing.features import NumericColumnSelector

bike_pipe = Pipeline([
    ('weekday_imputation', WeekdayImputer('weekday')),
    ('weathersit_imputation', WeathersitImputer('weathersit')),
    ('map_yr', Mapper('yr',config.model_config_.year_mapping)),
    ('map_mnth', Mapper('mnth',config.model_config_.month_mappings)),
    ('map_season', Mapper('season',config.model_config_.season_mappings)),
    ('map_weathersit', Mapper('weathersit',config.model_config_.weather_mappings)),
    ('map_holiday', Mapper('holiday',config.model_config_.holiday_mapping)),
    ('map_workingday', Mapper('workingday',config.model_config_.workingday_mapping)),
    ('map_hr', Mapper('hr',config.model_config_.hour_mapping)),
    ('outlier_handler', OutlierHandler(variables=['temp', 'atemp', 'hum', 'windspeed'], method='iqr', factor=1.5)),
    ('weekday_encoder', WeekdayOneHotEncoder(variables='weekday')),
    # Drop non-numeric columns
    ('numeric_selector', NumericColumnSelector()), 
    # # scale
    ('scaler', StandardScaler()),

    # Model fit
    ('model_rf', RandomForestRegressor(n_estimators=150, max_depth=5,random_state=42))
    ])
