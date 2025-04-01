import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from bikerental_model import __version__ as _version
from bikerental_model.config.core import config
from bikerental_model.pipeline import bike_pipe
from bikerental_model.processing.data_manager import load_pipeline
from bikerental_model.processing.data_manager import pre_pipeline_preparation
from bikerental_model.processing.validation import validate_inputs
import datetime


pipeline_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
#print(pipeline_file_name)
bikerental_pipeline= load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """
    input_df = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_df=input_df)
    results = {"predictions": None, "version": _version, "errors": errors}
    validated_data=validated_data.reindex(columns=config.model_config_.features)
    
    predictions = bikerental_pipeline.predict(validated_data)    
    results = {"predictions": predictions,"version": _version, "errors": errors}    

    print("Predictions", predictions)
    print("Errors", errors)

    return results

if __name__ == "__main__":

    #data_in={'dteday':["2012-11-05"],'season':["winter"],'hr':["2am"],'holiday':["No"],'weekday':["Mon"],
    #            'workingday':["Yes"],'weathersit':["Mist"],'temp':[6.10],'atemp':[3.0014],'hum':[49.0],'windspeed':[19.0012]}
    
    data_in = [
    {
      "dteday": "2012-11-05",
      "season": "winter",
      "hr": "2am",
      "holiday": "No",
      "weekday": "Mon",
      "workingday": "Yes",
      "weathersit": "Mist",
      "temp": 6.1,
      "atemp": 3.0014,
      "hum": 49,
      "windspeed": 19.0012
    }
  ]
    make_prediction(input_data=data_in)
