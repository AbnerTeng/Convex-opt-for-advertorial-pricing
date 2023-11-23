"""
Unit test code for preproc.py
"""
import os
import pandas as pd
import pytest
from src.preproc import PreProc

data = {
    'platform': ['fb', 'yt', 'ig'],
    'post_sum_6m': [23.0, 0.0, 74.0],
    'like_count_post_6m': [614.0, 0.0, 40946.0],
    'comment_count_post_6m': [23.0, 0.0, 3609.0],
    'share_count_post_6m': [16.0, 0.0, 0.0],
    'view_count_post_6m': [0.0, 0.0, 0.0],
    'video_post_sum_6m': [36.0, 39.0, 0.0],
    'like_count_video_6m': [2327.0, 21934.0, 0.0],
    'comment_count_video_6m': [79.0, 3556.0, 0.0],
    'share_count_video_6m': [248.0, 0.0, 0.0],
    'view_count_video_6m': [424308.0, 2055344.0, 0.0],
    'short_post_sum_6m': [0.0, 18.0, 20.0],
    'like_count_short_6m': [0.0, 581.0, 8921.0],
    'comment_count_short_6m': [0.0, 26.0, 667.0],
    'share_count_short_6m': [0.0, 0.0, 0.0],
    'view_count_short_6m': [0.0, 51482.0, 6416.0],
    'live_post_sum_6m': [0.0, 0.0, 0.0],
    'like_count_live_6m': [0.0, 0.0, 0.0],
    'comment_count_live_6m': [0.0, 0.0, 0.0],
    'share_count_live_6m': [0.0, 0.0, 0.0],
    'view_count_live_6m': [0.0, 0.0, 0.0],
    'follower_count_processed': [52000.0, 525000.0, 20000.0],
    'live': [25000.0, None, 17000.0],
    'post': [46000.0, None, 2000.0],
    'short': [187000.0, None, 88000.0],
    'video': [250000.0, 420000.0, None],
    'avg_post_voice': [5.91739, 0.0, 190.3824],
    'avg_video_voice': [1194.9749, 5493.5434, 0.0],
    'avg_short_voice': [0.0, 296.5611, 214.6500],
    'avg_live_voice': [0.0, 0.0, 0.0]
}
sample_df = pd.DataFrame(data)

@pytest.fixture
def preproc_instance() -> PreProc:
    """
    Initialize PreProc instance
    """
    instance = PreProc(f'{os.getcwd()}/data', False)
    instance.data = sample_df
    return instance

@pytest.mark.parametrize("platform, expected_length", [("fb", 1), ("yt", 1), ("ig", 1)])
def test_group_by_platform(platform, expected_length, preproc_instance) -> None:
    """
    Test the group_by_platform function
    """
    platform_data = preproc_instance.group_by_platform(platform)
    assert isinstance(platform_data, pd.DataFrame)
    assert len(platform_data) == expected_length


if __name__ == "__main__":
    pytest.main("-v", "test_preproc.py")
    