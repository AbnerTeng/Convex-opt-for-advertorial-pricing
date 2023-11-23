"""
unittest file for the property scoring function
"""
import os
import warnings
import pytest
import pandas as pd
from src.property_scoring import PropertyScoring
warnings.filterwarnings('ignore')

config = {
    'path': f'{os.getcwd()}/data/scoring_data_v3.csv',
    'w1': 0.167,
    'w2': 0.833
}

@pytest.fixture
def scoring_instance() -> PropertyScoring:
    """
    PropertyScoring instance
    """
    return PropertyScoring(**config)


def test_main(scoring_instance) -> None:
    """
    Check the main function
    """
    new_data = scoring_instance.main(
        scoring_instance.calculate_fod(
            scoring_instance.scaled_data
        )
    )
    assert isinstance(new_data, pd.DataFrame)
    assert all(
        0 <= new_data['score_stdev'].iloc[i] <= 1 for i in range(len(new_data['score_stdev']))
    )
    assert all(
        0 <= new_data['score_promo_exp'].iloc[i] <= 1 for i in range(len(new_data['score_promo_exp']))
    )
    assert min(new_data['rank']) == 1
    assert max(new_data['rank']) > 890
