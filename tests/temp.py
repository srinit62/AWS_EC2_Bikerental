from pydantic import BaseModel
from typing import Optional
from datetime import date
import pandas as pd

class Temp_Features(BaseModel):
    dteday: Optional[date]  # Date column
    season: Optional[str]  # Categorical (object)
    
data_in={'dteday':["2012-11-05"],
         'season':["winter"]}

df = pd.DataFrame.from_dict(data_in)

#temp_features = Temp_Features(df)
print(df)