import pandas as pd
import numpy as np
import io

def check_dataset_samples(df: pd.DataFrame) -> str:
    """
    Checks the first 10 rows of the dataset to understand each feature and whether the data is tabular or temporal.
    Returns a description string.
    """
    df_sample = df.head(10).to_string()
    buffer = io.StringIO()
    df.info(buf=buffer)
    df_info = buffer.getvalue()
         
    return f"""
    This is the first 10 rows of the dataset:
    {df_sample}
    
    This is the information of the dataset:
    {df_info}
    """
