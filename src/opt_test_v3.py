"""
Quadradic Programming optimization code of creation portfolio
version 0.4.0
Author: Andrew Lee, Abner Den

This script contains 3 classes:
- Attr
    This class contains heavy used attributes in constraints and optimization

- MyConstraints
    This class contains all constraints

- Optimizer
    This class is the full optimization process
"""
import time
from typing import List, Tuple, Dict
import numpy as np
from scipy.optimize import minimize
from .utils.data_utils import (
    sigmoid,
    load_data,
    to_df
)
from .utils.optim_utils import (
    cnt_transform,
    count_cost,
    count_voice,
    matrix_greed_search
)
from .constraints import MyConstraints
np.set_printoptions(
    precision=2, threshold=np.inf, suppress=True, linewidth=150
)


class Optimizer(MyConstraints):
    """
    Class of optimization

    ...

    New Arguments
    ---------
    label_path: str
        Path of preproc_data_v3.csv

    budget: int
        Firm's target budget

    voice: int
        Firm's target voice

    lamb: float
        Degree of penalty of either L1, L2, or Elastic Net regularization

    alpha: float
        Proportion of L1, L2 inside Elastic Net

    Methods
    -------
    |--- objective(__input__)
    |--- objective_l1(__input__)
    |--- objective_l2(__input__)
    └--- objective_enet(__input__):

    --> Objective functions with different regularization methods
    """

    def __init__(
        self, use_path: str, row: int,
        post_cnt_lst: List[int],
        spec_kols_lst: List[str],
        label_path: str,
        budget: int, voice: int, lamb: float, alpha: float
    ) -> None:
        """
        Matrix Parameters
        -----------------
        - use_data: shape = (row, col)

        - price_data: shape = (row * col/3, 1)

        - voice_data: shape = (row * col/3, 1)

        - label_data: shape = (row, 1)

        Other Parameters
        ----------------
        - count: int
            Iteration count

        - init_weight: list
            The copy of initial weight, used for adjusting weight

        - __lambda__: shape = (row * col, 1)
            Degree of penalty of either L1, L2, or Elastic Net regularization

        - __input__: shape = (row * col, 1)
            Input matrix of optimization
        """
        super(MyConstraints, self).__init__(
            use_path, row, post_cnt_lst, spec_kols_lst
        )
        self.label_path = label_path
        self.budget = budget
        self.voice = voice
        self.lamb = lamb
        self.alpha = alpha
        self.count = 0
        self.init_weight = self.weight.copy()
        self.label_data = np.expand_dims(
            np.array(load_data(self.label_path)['platform'])[:self.row], axis=1
        )
        self.__lambda__ = np.full((self.row * int(self.col / 3), 1), self.lamb)
        self.__input__ = np.array([0.1] * self.row * int(self.col / 3))

    def constraint_output(
        self, val: np.ndarray,
        trans_count: np.ndarray,
        input_count: np.ndarray
    ) -> float:
        """
        Add all result of constraints together

        Arguments
        ---------
        val: np.ndarray
            __input__ after reshape, shape = (row * col, 1)

        trans_count: np.ndarray
            Transformed count matrix, shape = (row, col / 3)

        input_count: np.ndarray
            __input__ matrix, shape = (row, col / 3)
        """
        if self.spec_kols_lst == ['None']:
            spec_kols = 0
        else:
            result = []
            for i in self.spec_kols_lst:
                semi_result = np.sum(input_count, axis=1)[int(i)]
                result.append(semi_result)
            spec_kols = [
                (result[i] - 1) * self.weight[6] for i in range(len(result))
            ]
        result = (self.budget - val.T @ self.price_data) * self.weight[0] + \
            (trans_count.T @ self.voice_data - self.voice) * self.weight[1] + \
            (
                np.sum(input_count, axis=0)[0] - self.post_cnt_lst[0]
            ) * self.weight[2] + \
            (
                np.sum(input_count, axis=0)[1] - self.post_cnt_lst[1]
            ) * self.weight[3] + \
            (
                np.sum(input_count, axis=0)[2] - self.post_cnt_lst[2]
            ) * self.weight[4] + \
            (
                np.sum(input_count, axis=0)[3] - self.post_cnt_lst[3]
            ) * self.weight[5] + \
            np.sum(spec_kols)
        return result.item()

    def objective(self, __input__: List[float], lamb=None) -> float:
        """
        Parameters:
        ===========================
        - __input__: shape = (row * col, )
        ===========================
        Matrix multiplication:

                            [p0,
                             p1,
        [x0, x1, ..., xn] @  p2,
                             ...,
                             pn]
        """
        val = np.expand_dims(__input__, 1)
        trans_count = np.expand_dims(
            cnt_transform(
                __input__, self.row, int(self.col / 3), self.k
            ), 1
        )
        input_cnt = val.reshape(self.row, int(self.col / 3))
        obj_opt = val.T @ self.price_data
        final_opt = self.constraint_output(val, trans_count, input_cnt)
        return final_opt + obj_opt.item()

    def objective_l1(
        self,
        __input__: List[float],
        lamb: List[float]
    ) -> float:
        """
        Parameters:
        ===========================
        - __input__: shape = (row * col, )
        - lamb: shape = (row * col, 1)
        """
        val = np.expand_dims(__input__, 1)
        trans_count = np.expand_dims(
            cnt_transform(
                __input__, self.row, int(self.col / 3), self.k
            ), 1
        )
        input_cnt = val.reshape(self.row, int(self.col / 3))
        obj_opt = val.T @ self.price_data
        final_opt = self.constraint_output(val, trans_count, input_cnt)
        __input__ = np.expand_dims(__input__, 1)
        l1_term = lamb.T @ np.abs(__input__)
        return final_opt + l1_term.item() + obj_opt.item()

    def objective_l2(
        self,
        __input__: List[float],
        lamb: List[float]
    ) -> float:
        """
        Parameters:
        ===========================
        - __input__: shape = (row * col, )
        - lamb: shape = (row * col, 1)
        """
        val = np.expand_dims(__input__, 1)
        trans_count = np.expand_dims(
            cnt_transform(
                __input__, self.row, int(self.col / 3), self.k
            ), 1
        )
        input_cnt = val.reshape(self.row, int(self.col / 3))
        obj_opt = val.T @ self.price_data
        final_opt = self.constraint_output(val, trans_count, input_cnt)
        __input__ = np.expand_dims(__input__, 1)
        l2_term = lamb.T @ np.square(__input__)
        return final_opt + (l2_term.item() / 2) + obj_opt.item()

    def objective_enet(
        self,
        __input__: List[float],
        lamb: List[float]
    ) -> float:
        """
        Parameters:
        ===========================
        - __input__: shape = (row * col, )
        - lamb: shape = (row * col, 1)
        """
        val = np.expand_dims(__input__, 1)
        trans_count = np.expand_dims(
            cnt_transform(
                __input__, self.row, int(self.col / 3), self.k
            ), 1
        )
        input_cnt = val.reshape(self.row, int(self.col / 3))
        obj_opt = val.T @ self.price_data
        final_opt = self.constraint_output(val, trans_count, input_cnt)
        __input__ = np.expand_dims(__input__, 1)
        l1_term = lamb.T @ np.abs(__input__)
        l2_term = lamb.T @ np.square(__input__)
        return final_opt + obj_opt.item() + \
            self.alpha * l1_term.item() \
            + ((1 - self.alpha) / 2) * l2_term.item()

    def get_functions(self, types: str) -> Tuple:
        """
        Get functions

        Parameters
        ----------
        - types: type of objective function

        Output
        ------
        - func: objective function
        - cons_func: constraints function
        """
        if types == 'plain':
            func = self.objective
        elif types == 'l1':
            func = self.objective_l1
        elif types == 'l2':
            func = self.objective_l2
        elif types == 'enet':
            func = self.objective_enet
        cons_func = [
            {'type': 'ineq', 'fun': self.cost_cons, 'args': (self.budget, )},
            {'type': 'ineq', 'fun': self.voice_cons, 'args': (self.voice, )},
            {'type': 'ineq', 'fun': self.live_cnt},
            {'type': 'ineq', 'fun': self.post_cnt},
            {'type': 'ineq', 'fun': self.short_cnt},
            {'type': 'ineq', 'fun': self.vid_cnt},
            {'type': 'ineq', 'fun': self.spec_kols}
        ]
        return func, cons_func

    def run_opt(self, func, cons_func: Dict) -> tuple:
        """
        Run the full optimization process

        Parameters
        ----------
        - func: objective function
        - cons_func: constraints function

        Output
        ------
        - solution: the full information and result of the optimization
        - solution.x: the optimized __init__ matrix
        """
        print("Optimization start!")
        print("===================================================")
        start_time = time.time()
        solution = minimize(
            fun=func,
            x0=self.__input__,
            args=([0] if func is self.objective else (self.__lambda__, )),
            method='SLSQP',
            bounds=[(0, 100)] * self.row * int(self.col / 3),
            constraints=cons_func,
            options={'maxiter': 500, 'disp': True}
        )
        end_time = time.time()
        print("===================================================")
        print("End of optimization")
        print(f"Time elapsed: {end_time - start_time} seconds")
        return solution, solution.x

    def get_opt_result(self, solution) -> Tuple:
        """
        Print optimization result

        Arguments
        ---------
        - solution: the full information and result of the optimization

        * We only need the matrix part of the solution

        Output
        ------
        - c_cost: optimized cost
        - c_voice: optimized voice
        - result_with_label: optimized matrix with label
        """
        count_result = np.reshape(
            solution.x, (int(self.row), int(self.col / 3))
        )
        final_result = np.concatenate(
            (
                count_result,
                np.reshape(
                    self.price_data, (
                        int(self.row), int(self.col / 3)
                    )
                ),
                np.reshape(
                    self.voice_data, (
                        int(self.row), int(self.col / 3)
                    )
                )
            ),
            axis=1
        )
        final_result = matrix_greed_search(final_result)
        result_with_label = np.concatenate(
            (self.label_data, final_result),
            axis=1
        )
        c_cost = count_cost(final_result, self.col)
        c_voice = count_voice(final_result, self.col)
        print(f"Output Cost: {c_cost} (Thousand dollars)")
        print(f"Output Voice: {c_voice}")
        return c_cost, c_voice, result_with_label

    def adjust_weight(
        self, obj_function, constraints: Dict, budget: float, voice: float
    ) -> Tuple:
        """
        Adjust weight automatically based on the result of optimization

        Arguments
        ---------
        - obj_function:
            objective function we use

        - constraints:
            constraints we use

        - budget:
            current budget

        - voice:
            current voice

        Output
        ------
        Same as the run_opt() function
        """
        self.count += 1

        if (budget < self.budget and voice > self.voice) or (self.count > 200):
            solution, _ = self.run_opt(obj_function, constraints)
            print("Optimization terminated")
            _, _, result_with_label = self.get_opt_result(solution)
            print(to_df(result_with_label))
            return result_with_label


        if (
            (budget > self.budget and voice < self.voice) or \
            (budget < self.budget and voice < self.voice) or \
            (budget > self.budget and voice > self.voice)
        ):
            print(
                f"Iterate: {self.count}. Constraint not satisfied, start adjusting constrant weight"
            )
            cons_result = []
            for cons in constraints:
                if cons['fun'].__name__ == "cost_cons":
                    cons_result.append(cons['fun'](self.__input__, self.budget))
                elif cons['fun'].__name__ == "voice_cons":
                    cons_result.append(cons['fun'](self.__input__, self.voice))
                elif cons['fun'].__name__ == "spec_kols":
                    cons_result.append(np.sum(cons['fun'](self.__input__)))
                else:
                    cons_result.append(cons['fun'](self.__input__))

            cons_result = [sigmoid(np.abs(cons)) for cons in cons_result]
            mean_weight, sum_cons = np.mean(self.init_weight), np.sum(cons_result)

            for idx, _ in enumerate(self.weight):
                multiplier = (sum_cons - cons_result[idx]) / sum_cons

                if self.weight[idx] < mean_weight:
                    self.weight[idx] = self.weight[idx] * 0.77 * multiplier

                if self.weight[idx] < 0.01:
                    self.weight[idx] = 0

            print(f"Iterate: {self.count}. Adjusted weight: {self.weight}")
            solution, _ = self.run_opt(obj_function, constraints)
            comp_cost, comp_voice, result_with_label = self.get_opt_result(solution)
            print(to_df(result_with_label))
            result_with_label = self.adjust_weight(
                obj_function, constraints, comp_cost, comp_voice
            )
            return result_with_label

    def print_final_output(self, types: str) -> None:
        """
        Print final output
        
        Arguments
        ---------
        - types: type of objective function
        """
        print("============== Optimize Done, output information ===============")
        print(f"Selected rows: {self.row} rows")
        print(f"Objective function types: {types}")
        print(f"Budget Constraint: {self.budget} (Thousand Dollars)")
        print(f"Target Voice: {self.voice}")
        if types in ['l1', 'l2', 'enet']:
            print(f"Regularization lambda: {self.lamb}")
        if types == 'enet':
            print(f"Elastic Net alpha: {self.alpha}")
        print(
            f"Minimum count: Live: {self.post_cnt_lst[0]}, " + \
            f"Post: {self.post_cnt_lst[1]}, " + \
            f"Short: {self.post_cnt_lst[2]}, " + \
            f"Vid: {self.post_cnt_lst[3]}"
        )
        print(f"Choosed KOLs: KOL {self.spec_kols_lst}")
