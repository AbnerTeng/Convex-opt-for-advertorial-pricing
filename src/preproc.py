"""
Data preprocessor

This script allows the user to preprocess the raw data of KOL post type
and their related price.

This tool accepts a path to the raw data and a boolean argument to download
"""
import os
import warnings
import pandas as pd
from .utils import load_data
warnings.filterwarnings('ignore')


class PreProc:
    """
    A class used for data preprocessing
    
    ...
    
    Attributes
    ----------
    path: str
        path to the data folder

    data: pd.DataFrame
        raw data to be processed

    arg: bool
        argument to download data or not
    """
    def __init__(self, dat_name: str, arg: bool) -> None:
        """
        Parameters
        ----------
        path: str
            path to the data folder
        data: pd.DataFrame
            raw data to be processed
        arg: bool
            argument to download data or not
        """
        self.dat_name = dat_name
        self.data = load_data(f"{os.getcwd()}/data/{self.dat_name}.csv")
        self.arg = arg


    def interpolate(self) -> None:
        """
        Interpolate missing values
        """
        self.data = self.data.fillna(5000000)


    def filter_cols(self) -> pd.DataFrame:
        """
        Filtering columns
        """
        use_data = self.data[
            [
                'live_post_sum_6m', 'post_sum_6m', 'short_post_sum_6m', 'video_post_sum_6m',
                'live', 'post', 'short', 'video', 'avg_live_voice', 'avg_post_voice',
                'avg_short_voice', 'avg_video_voice'
            ]
        ]
        return use_data


    def for_raw_data(self) -> None:
        """
        preprocessing procedure for raw data
        """
        self.interpolate()
        use_data = self.filter_cols()
        if self.arg:
            self.data.to_csv(
                f'{os.getcwd()}/data/preproc_data_v3.csv', encoding = 'utf-8', index = False
            )
            use_data.to_csv(
                f'{os.getcwd()}/data/use_data_v2.csv', encoding = 'utf-8', index = False
            )


    def for_preproc_data(self) -> None:
        """
        preprocessing procedure for preprocessed data
        """
        use_data = self.filter_cols()
        if self.arg:
            use_data.to_csv(
                f'{os.getcwd()}/data/use_data_v2.csv', encoding = 'utf-8', index = False
            )
    