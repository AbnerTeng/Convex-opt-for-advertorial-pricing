from typing import Union, Dict, List
import json
import yaml
import numpy as np
import pandas as pd


def sigmoid(num: float) -> float:
    """
    sigmoid function
    
    Parameters
    ----------
    x: float
        input value
    """
    return 1 / (1 + np.exp(-num))


def load_data(path: str) -> Union[pd.DataFrame, Dict, None]:
    """
    Load files from specific folder.
    
    Parameters
    ----------
    path: str
        path of folder
    """
    suffix = path.split('.')[-1]
    if suffix is None:
        return None
    if suffix == 'csv':
        data = pd.read_csv(path, encoding = 'utf-8')
    elif suffix == "json":
        with open(path, 'r', encoding = 'utf-8') as json_file:
            data = json.load(json_file)
    elif suffix == "yaml":
        with open(path, 'r', encoding = 'utf-8') as yaml_file:
            data = yaml.load(yaml_file, Loader = yaml.FullLoader)
    else:
        raise ValueError("Unsupported file format")
    return data


def simple_trans(mat: np.ndarray, k: float) -> List[float]:
    """
    simple voice decrease function
    
    Parameters
    ----------
    mat: np.ndarray
        initial post_count data, size: (rows * cols, )
    k: float
        Degree of descent for specific post type
    """
    ratio = np.exp(-1/k)
    return [
        0 if element == 0 \
        else ((1 - ratio ** element) / (1 - ratio)) \
        for element in mat
    ]


def min_max_scaler(mat: np.ndarray) -> np.ndarray:
    """
    Min-Max Scaler
    
    Parameters
    ----------
    mat: np.ndarray
        input data
    """
    min_arr, max_arr = np.min(mat, axis=0), np.max(mat, axis=0)
    return (mat - min_arr) / (max_arr - min_arr)


def inverse_min_max_scaler(mat: np.ndarray) -> np.ndarray:
    """
    Inverse Min-Max Scaler
    
    Parameters
    ----------
    mat: np.ndarray
        input data
    """
    min_arr, max_arr = np.min(mat, axis=0), np.max(mat, axis=0)
    return -(mat - max_arr) / (max_arr - min_arr)


def standard_scaler(mat: np.ndarray) -> np.ndarray:
    """
    Standard Scaler
    
    Parameters
    ----------
    mat: np.ndarray
        input data
    """
    mean_arr, std_arr = np.mean(mat, axis=0), np.std(mat, axis=0)
    return (mat - mean_arr) / std_arr


def normalization(mat: np.ndarray, norm_type: str) -> np.ndarray:
    """
    normalization method
    1. min-max scaler
    2. inverse_min-max scaler
    3. z-score scaler
    
    ----------
    norm_type: str
        - min-max
        - inverse_min-max
        - standard
    """
    if norm_type == "min-max":
        scaler = min_max_scaler
    elif norm_type == "inverse_min-max":
        scaler = inverse_min_max_scaler
    else:
        scaler = standard_scaler
    return scaler(mat)


def to_df(mat: np.ndarray) -> pd.DataFrame:
    """
    Transform np.ndarray to pd.DataFrame
    
    Parameters
    ----------
    mat: np.ndarray
        input matrix
    """
    return pd.DataFrame(
        mat, columns=[
            'platform', 'live_count', 'post_count', 'short_count', 'vid_count',
            'live_price', 'post_price', 'short_price', 'vid_price',
            'live_voice', 'post_voice', 'short_voice', 'vid_voice'
        ]
    )
