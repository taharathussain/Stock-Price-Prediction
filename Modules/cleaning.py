import pandas as pd
import numpy as np
from scipy import stats

# Check for missing values
def missing_values(data):
    missing_values = data.isnull().sum()
    return missing_values

# Check for duplicates
def duplicates(data):
    duplicate_rows = data.duplicated().sum()
    return duplicate_rows

# Remove duplicates
def remove_duplicates(data):
    data.drop_duplicates(inplace=True)
    return data
