import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
from datetime import date

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from bikerental_model.config.core import config
from bikerental_model.processing.data_manager import pre_pipeline_preparation


def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""
    pre_processed = pre_pipeline_preparation(data_frame=input_df)
    validated_data = pre_processed[config.model_config_.features].copy()
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    validated_data = pre_processed[config.model_config_.features].copy()

    return validated_data, errors


class DataInputValidationSchema(BaseModel):
    dteday: Optional[date]  # Date column
    season: Optional[str]  # Categorical (object)
    hr: Optional[str]  # Categorical (object)
    holiday: Optional[str]  # Categorical (object)
    weekday: Optional[str]  # Categorical (object) - has missing values
    workingday: Optional[str]  # Categorical (object)
    weathersit: Optional[str]  # Categorical (object) - has missing values
    temp: Optional[float]  # Continuous numeric
    atemp: Optional[float]  # Continuous numeric
    hum: Optional[float]  # Continuous numeric
    windspeed: Optional[float]  # Continuous numeric
    yr: Optional[str]  # Categorical (object)
    mnth: Optional[str]  # Categorical (object)


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputValidationSchema]
