"""
General utilities for creation portfolio

This script has utilities adapt to several files and specific to optimization procedures
"""
import math
import pandas as pd
import numpy as np
from tqdm import tqdm


def sigmoid(x: float) -> float:
    """
    sigmoid function
    
    Parameters
    ----------
    x: float
        input value
    """
    return 1 / (1 + math.exp(-x))


def load_data(path: str) -> pd.DataFrame:
    """
    Load .csv file from specific folder.
    
    Parameters
    ----------
    path: str
        path of folder
    """
    data = pd.read_csv(path, encoding = 'utf-8')
    return data


def simple_trans(mat: np.ndarray, k: float) -> list:
    """
    simple voice decrease function
    
    Parameters
    ----------
    mat: np.ndarray
        initial post_count data, size: (rows * cols, )
    k: float
        Degree of descent for specific post type
    """
    ratio = math.exp(-1/k)
    return [
        0 if element == 0 \
        else ((1 - ratio ** element) / (1 - ratio)) \
        for element in mat
    ]


def cnt_transform(post_cnt: list, rows: int, cols: int, k: list) -> list:
    """
    voice descent function for each post type
    
    Parameters
    ----------
    post_cnt: list
        initial post_count data, size: (rows * cols, )
    rows: int
        Number of chosen KOLs (should be args.row)
    cols: int
        Number of post_types (should be 4)
    k: list
        Degree of descents for specific post type
    """
    post_cnt = np.expand_dims(post_cnt, axis = 1).reshape(rows, cols).T
    temp = np.zeros((cols, rows))
    for col, _ in enumerate(temp):
        ratio = math.exp(-1 / k[col])
        for row, element in enumerate(post_cnt[col]):
            if element == 0:
                continue
            else:
                temp[col, row] = (1 - ratio ** element) / (1 - ratio)
    return temp.T.flatten()


def count_cost(mat: np.ndarray, cols: int) -> float:
    """
    Calculate and check if the total cost exceed the firm's budget
    
    Parameters
    ----------
    mat: np.ndarray
        Final result of optimization, size: (args.row, 12)
    cols: int
        Amount of columns
    """
    result = np.sum(
        [mat[:, i] * mat[:, i + int(cols/3)] for i in range(int(cols/3))]
    )
    return result


def count_voice(mat: np.ndarray, cols: float) -> float:
    """
    Calculate and check if the total voice exceed the firm's target voice
    
    Parameters
    ----------
    mat: np.ndarray
        Final result of optimization, size: (args.row, 12)
    cols: int
        Amount of columns
    """
    result = np.sum(
        [mat[:, i] * mat[:, i + 2 * int(cols/3)] for i in range(int(cols/3))]
    )
    return result


def check_valid_cost(cost: float, target: int) -> None:
    """
    Check if the total cost exceed the firm's budget
    
    Parameters
    ----------
    cost: float
        The return value of function count_cost(), typically the total cost after
        optimization
    target: int
        The budget constraint of firm
    
    Raises
    ------
    ValueError
        If the output cost is higher than the target cost
    """
    print("=========== Message ===========")
    if cost > target:
        raise ValueError('Budget constraint is not satisfied')
    else:
        print('Budget constraint is satisfied')


def check_valid_voice(voice: float, target: int) -> None:
    """
    Check if the total voice exceed the firm's target voice
    
    Parameters
    ----------
    voice: float
        The return value of function count_voice(), typically the total voice after
        optimization
    target: int
        The target voice of firm
    
    Raises
    ------
    ValueError
        If the output voice is less than the target voice
    """
    print("=========== Message ===========")
    if voice < target:
        raise ValueError('Target voice constraint is not satisfied')
    else:
        print('Target voice constraint is satisfied')


def get_specific_matrices(mat: np.ndarray) -> np.ndarray:
    """
    Get matrices of count, price, voice, vp_ratio
    
    Parameters
    ----------
    mat: np.ndarray
        Final result of optimization, size: (args.row, 12)
    """
    count_mat = mat[:, :int(mat.shape[1]/3)]
    price_mat = mat[:, int(mat.shape[1]/3): 2 * int(mat.shape[1]/3)]
    voice_mat = mat[:, 2 * int(mat.shape[1]/3):]
    vp_ratio_mat = np.divide(voice_mat, price_mat)
    return count_mat, price_mat, voice_mat, vp_ratio_mat


def matrix_greed_search(matrix: np.ndarray) -> np.ndarray:
    """
    Greed search of the optimal matrix with the highest voice-price ratio
    
    Parameters
    ----------
    matrix: np.ndarray 
        Final result of optimization, size: (args.row, 12)
    """
    rows, cols = matrix.shape
    count_mat, price_mat, voice_mat, vp_ratio_mat = get_specific_matrices(matrix)
    count_1d, price_1d, voice_1d, vp_1d = \
    count_mat.flatten(), price_mat.flatten(), voice_mat.flatten(), vp_ratio_mat.flatten()
    count_vp_mat = np.array([count_1d, price_1d, voice_1d, vp_1d])
    sorted_indeces = np.argsort(vp_1d)[::-1]
    org_indeces = np.argsort(sorted_indeces)
    count_1d = count_1d[sorted_indeces]
    sorted_data = count_vp_mat[:, sorted_indeces]
    max_cp = np.sum(sorted_data[0] * sorted_data[2]) / np.sum(sorted_data[0] * sorted_data[1])
    voice_adj = 0
    init_voice = np.sum(sorted_data[0] * sorted_data[2])
    init_price = np.sum(sorted_data[0] * sorted_data[1])
    for i in tqdm(range(sorted_data.shape[1])):
        if sorted_data[0, i] > 0.1:
            if max_cp < sorted_data[3, i]:
                sorted_data[0, i] = np.ceil(sorted_data[0, i])
                adj_count = sorted_data[0, i] - count_1d[i]
                current_cp = (init_voice + adj_count * sorted_data[2, i]) / (init_price + adj_count * sorted_data[1, i])
                max_cp = current_cp
                voice_adj += adj_count * sorted_data[2, i]
            else:
                sorted_data[0, i] = np.floor(count_1d[i])
                adj_count = count_1d[i] - sorted_data[0, i]
                voice_adj -= adj_count * sorted_data[2, i]
                if voice_adj < 0:
                    sorted_data[0, i] = np.ceil(count_1d[i])
                    voice_adj += sorted_data[2, i]
        else:
            voice_adj -= sorted_data[0, i] * sorted_data[2, i]
            sorted_data[0, i] = 0

    org_data = sorted_data[:, org_indeces]
    result = np.concatenate(
        (
            org_data[0].reshape(rows, int(cols/3)),
            org_data[1].reshape(rows, int(cols/3)),
            org_data[2].reshape(rows, int(cols/3))
        ),
        axis = 1
    )
    return result


def min_max_scaler(mat: np.ndarray) -> np.ndarray:
    """
    min_max_scaler
    
    Arguments
    ---------
    mat: np.ndarray
        input matrix
    """
    min_arr, max_arr = np.min(mat, axis = 0), np.max(mat, axis = 0)
    result = (mat - min_arr) / (max_arr - min_arr)
    return result


def inverse_min_max_scaler(mat: np.ndarray) -> np.ndarray:
    """
    inverse min_max_scaler
    
    Arguments
    ---------
    mat: np.ndarray
        input matrix
    """
    min_arr, max_arr = np.min(mat, axis = 0), np.max(mat, axis = 0)
    result = -(mat - max_arr) / (max_arr - min_arr)
    return result


def standard_scaler(mat: np.ndarray) -> np.ndarray:
    """
    standard_scaler
    
    Arguments
    ---------
    mat: np.ndarray
        input matrix
    """
    mean_arr, std_arr = np.mean(mat, axis = 0), np.std(mat, axis = 0)
    result = (mat - mean_arr) / std_arr
    return result


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
