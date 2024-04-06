"""
constraints
"""
from typing import List
import numpy as np
from .utils.data_utils import load_data
from .utils.optim_utils import cnt_transform
from .constants import DEFAULT_K


class Attr:
    """
    Heavy using attributes

    Arguments
    ---------
    - use_path: str
        path of use data
    - row: int
        number of rows to be optimized(number of KOL to be selected)

    - post_cnt_lst: list
        post count constraint for each type, order: live, post, short, vid

    - spec_kols_lst: list
        specific kols need to be selected

    Attributes
    ----------
    - use_data: pd.DataFrame
        full data used for optim, contains all KOLs promotion price and voice

    - row: int
        number of rows to be optimized(number of KOL to be selected)

    - col: int
        number of columns of use data

    - price_data: np.ndarray
        promotion price data of use data

    - voice_data: np.ndarray
        voice data of use data

    - weight: list
        Initial weight of each constraint, the firm will customize it

    - k: list
        Degree of voice descent of specific post type

    - post_cnt_lst: list
        post count constraint for each type, order: live, post, short, vid

    - spec_kols_lst: list
        specific kols need to be selected
    """

    def __init__(
        self, use_path: str, row: int,
        post_cnt_lst: List[int],
        spec_kols_lst: List[str]
    ) -> None:
        self.use_data = np.array(
            load_data(use_path)
        )
        self.row = row
        self.col = len(self.use_data[0])
        self.price_data = (
            self.use_data[
                :self.row, int(self.col/3): 2 * int(self.col/3)
            ] / 1000
        ).reshape(self.row * int(self.col/3), 1)
        self.voice_data = (
            self.use_data[:self.row, 2 * int(self.col/3):]
        ).reshape(
            self.row * int(self.col/3), 1
        )
        self.weight = [0.9, 0.8, 0.2, 0.2, 0.2, 0.2, 0.3]
        self.k = DEFAULT_K
        self.post_cnt_lst = post_cnt_lst
        self.spec_kols_lst = spec_kols_lst

class MyConstraints(Attr):
    """
    Class of all constraints, the father class is Attr, which contains all
    attributes that will be used in constraints

    ...

    constraints:
    - constraint_cost(__input__, budget)
        Lower the cost within the budget

    - constraint_voice(_input__, voice)
        Raise the voice beyond the target voice

    - constraint_live_cnt(__input__)
        Let live post count be greater than the minimum count

    - constraint_post_cnt(__input__)
        Let normal post count be greater than the minimum count

    - constraint_short_cnt(__input__)
        Let short count be greater than the minimum count

    - constraint_vid_cnt(__input___)
        Let video count be greater than the minimum count

    - constraint_spec_kols(__input__)
        Let specific KOLs be greater than the minimum count
    """

    def __init__(
        self, use_path: str, row: int,
        post_cnt_lst: List[int],
        spec_kols_lst: List[str]
    ) -> None:
        super(Attr, self).__init__(use_path, row, post_cnt_lst, spec_kols_lst)

    def cost_cons(self, __input__: List[float], budget: int) -> float:
        """
        Parameters:
        ==========================
        - __input__: shape = (row * col, )
        - budget: shape = (1, )
        """
        val = np.expand_dims(__input__, 1)
        result = budget - val.T @ self.price_data
        return result.item() * self.weight[0]

    def voice_cons(self, __input__: List[float], voice: int) -> float:
        """
        Parameters:
        ==========================
        - __input__: shape = (row * col, )
        - voice: shape = (1, )
        """
        trans_count = cnt_transform(
            __input__, self.row, int(self.col / 3),
            self.k
        )
        trans_count = np.expand_dims(trans_count, 1)
        result = trans_count.T @ self.voice_data - voice
        return result.item() * self.weight[1]

    def live_cnt(self, __input__: List[float]) -> float:
        """
        Parameters:
        ==========================
        - __input__: shape = (row * col, )
        """
        __input__ = np.expand_dims(
            __input__, 1
        ).reshape(self.row, int(self.col / 3))
        result = np.sum(__input__, axis=0)[0]
        return (result - self.post_cnt_lst[0]) * self.weight[2]

    def post_cnt(self, __input__: List[float]) -> float:
        """
        Parameters:
        ==========================
        - __input__: shape = (row * col, )
        """
        __input__ = np.expand_dims(
            __input__, 1
        ).reshape(self.row, int(self.col / 3))
        result = np.sum(__input__, axis=0)[1]
        return (result - self.post_cnt_lst[1]) * self.weight[3]

    def short_cnt(self, __input__: List[float]) -> float:
        """
        Parameters:
        ==========================
        - __input__: shape = (row * col, )
        """
        __input__ = np.expand_dims(
            __input__, 1
        ).reshape(self.row, int(self.col / 3))
        result = np.sum(__input__, axis=0)[2]
        return (result - self.post_cnt_lst[2]) * self.weight[4]

    def vid_cnt(self, __input__: List[float]) -> float:
        """
        Parameters:
        ==========================
        - __input__: shape = (row * col, )
        """
        __input__ = np.expand_dims(
            __input__, 1
        ).reshape(self.row, int(self.col / 3))
        result = np.sum(__input__, axis=0)[3]
        return (result - self.post_cnt_lst[3]) * self.weight[5]

    def spec_kols(self, __input__: List[float]) -> list:
        """
        Parameters:
        ===========================
        - __input__: shape = (row * col, )
        """
        if self.spec_kols_lst == ['None']:
            return 0
        else:
            __input__ = np.expand_dims(
                __input__, 1
            ).reshape(self.row, int(self.col / 3))
            result = []
            for i in self.spec_kols_lst:
                result.append(np.sum(__input__, axis=1)[int(i)])
            return [
                (result[i] - 1) * self.weight[6] for i in range(len(result))
            ]
