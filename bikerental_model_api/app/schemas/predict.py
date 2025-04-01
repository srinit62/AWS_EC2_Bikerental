from typing import Any, List, Optional

from pydantic import BaseModel, StrictStr
from datetime import date


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[float]

class DataInputSchemaValidation(BaseModel):
    dteday: Optional[date]  # Date column
    season: Optional[str] = None  # Categorical (object)
    hr: Optional[str] = None # Categorical (object)
    holiday: Optional[str] = None # Categorical (object)
    weekday: Optional[str] = None # Categorical (object) - has missing values
    workingday: Optional[str] = None # Categorical (object)
    weathersit: Optional[str] = None # Categorical (object) - has missing values
    temp: Optional[float] = None # Continuous numeric
    atemp: Optional[float] = None # Continuous numeric
    hum: Optional[float] = None # Continuous numeric
    windspeed: Optional[float] = None # Continuous numeric
    mnth: Optional[str] = None # Categorical (object)

class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchemaValidation]

class Item(BaseModel):
    inputs: List[DataInputSchemaValidation]

