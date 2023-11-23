"""
Data preprocessor

This script allows the user to preprocess the raw data of KOL post type
and their related price.

This tool accepts a path to the raw data and a boolean argument to download
"""
import os
import warnings
from argparse import ArgumentParser
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
        
    Methods
    -------
    group_by_platform(platform: str) -> pd.DataFrame:
        split data by social media platform. ex: fb, ig, yt
    main() -> None:
        main execute code
    """
    def __init__(self, path: str, arg: bool) -> None:
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
        self.path = path
        self.data = load_data(f"{self.path}/opt_data_v2.csv")
        self.arg = arg


    def group_by_platform(self, platform: str) -> pd.DataFrame:
        """
        split data by social media platform. ex: fb, ig, yt
        
        Parameters
        ----------
        platform: str
            social media platform
        """
        platform_group = self.data.groupby('platform')
        return platform_group.get_group(platform)


    def main(self) -> None:
        """
        Main execute code
        """
        self.data = self.data.fillna(5000000)
        fb_data = self.group_by_platform('fb')
        yt_data = self.group_by_platform('yt')
        ig_data = self.group_by_platform('ig')
        use_data = self.data[['live_post_sum_6m', 'post_sum_6m', 'short_post_sum_6m', 'video_post_sum_6m',
                              'live', 'post', 'short', 'video',
                              'avg_live_voice', 'avg_post_voice', 'avg_short_voice', 'avg_video_voice'
                            ]]
        if self.arg:
            self.data.to_csv(f'{self.path}/preproc_data_v3.csv', encoding = 'utf-8', index = False)
            fb_data.to_csv(f'{self.path}/fb_data.csv', encoding = 'utf-8', index = False)
            yt_data.to_csv(f'{self.path}/yt_data.csv', encoding = 'utf-8', index = False)
            ig_data.to_csv(f'{self.path}/ig_data.csv', encoding = 'utf-8', index = False)
            use_data.to_csv(f'{self.path}/use_data.csv', encoding = 'utf-8', index = False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '-d', '--download',
        type = bool, default = False,
        help = 'Download data to local folder or not'
    )
    args = parser.parse_args()
    data_path = f'{os.getcwd()}/data'
    preproc = PreProc(data_path, args.download)
    preproc.main()
    