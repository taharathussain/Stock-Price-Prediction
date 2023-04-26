# Loading libraries
import pandas  as pd
import numpy as np


def dataload(file_path):
    """Load dataset from a file."""
    data = pd.read_csv(file_path)
    return data
