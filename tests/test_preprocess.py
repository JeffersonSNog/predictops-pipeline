import pytest
import pandas as pd
from src.preprocess import engineer_features, encode_type



def test_encode_type():
    df = pd.DataFrame({"Type": ["L", "M", "H"]})
    result = encode_type(df)
    assert list(result["Type"]) == [0, 1, 2]

def test_engineer_features():
    df = pd.DataFrame({'Process temperature [K]': [1, 2, 3],
                       'Air temperature [K]': [1, 2, 3],
                       'Rotational speed [rpm]': [1, 2, 3],
                       'Torque [Nm]': [1, 2, 3],
                       'Tool wear [min]': [1, 2, 3]})
    result = engineer_features(df)
    assert list(result['temp_diff']) == [0, 0, 0]
    assert list(result['power']) == [1, 4, 9]
    assert list(result['tool_wear_torque']) == [1, 4, 9]