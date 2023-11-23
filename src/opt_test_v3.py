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
import numpy as np
from scipy.optimize import minimize
from .utils import (
    sigmoid,
    load_data,
    cnt_transform,
    count_cost,
    count_voice,
    matrix_greed_search,
    to_df
)
np.set_printoptions(precision=2, threshold=np.inf, suppress=True, linewidth=150)
DEFAULT_K = [2.6, 2.8, 3.0, 3.2]


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
        full data used by optimization, contains all KOLs promotion price and voice
    
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
        self, use_path: str, row: int, post_cnt_lst: list, spec_kols_lst: list
    ) -> None:
        self.use_data = np.array(
            load_data(use_path)
        )
        self.row = row
        self.col = len(self.use_data[0])
        self.price_data = (
            self.use_data[:self.row, int(self.col/3): 2 * int(self.col/3)] / 1000
        ).reshape(self.row * int(self.col/3), 1)
        self.voice_data = (
            self.use_data[:self.row, 2 * int(self.col/3):]).reshape(self.row * int(self.col/3), 1
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
            self, use_path: str, row: int, post_cnt_lst: list, spec_kols_lst: list
        ) -> None:
        super(Attr, self).__init__(use_path, row, post_cnt_lst, spec_kols_lst)


    def cost_cons(self, __input__: list, budget: int) -> float:
        """
        Parameters:
        ==========================
        - __input__: shape = (row * col, )
        - budget: shape = (1, )
        """
        val = np.expand_dims(__input__, 1)
        result = budget - val.T @ self.price_data
        return result.item() * self.weight[0]


    def voice_cons(self, __input__: list, voice: int) -> float:
        """
        Parameters:
        ==========================
        - __input__: shape = (row * col, )
        - voice: shape = (1, )
        """
        trans_count = cnt_transform(__input__, self.row, int(self.col / 3), self.k)
        trans_count = np.expand_dims(trans_count, 1)
        result = trans_count.T @ self.voice_data - voice
        return result.item() * self.weight[1]


    def live_cnt(self, __input__: list) -> float:
        """
        Parameters:
        ==========================
        - __input__: shape = (row * col, )
        """
        __input__ = np.expand_dims(__input__, 1).reshape(self.row, int(self.col / 3))
        result = np.sum(__input__, axis = 0)[0]
        return (result - self.post_cnt_lst[0]) * self.weight[2]


    def post_cnt(self, __input__: list) -> float:
        """
        Parameters:
        ==========================
        - __input__: shape = (row * col, )
        """
        __input__ = np.expand_dims(__input__, 1).reshape(self.row, int(self.col / 3))
        result = np.sum(__input__, axis = 0)[1]
        return (result - self.post_cnt_lst[1]) * self.weight[3]


    def short_cnt(self, __input__: list) -> float:
        """
        Parameters:
        ==========================
        - __input__: shape = (row * col, )
        """
        __input__ = np.expand_dims(__input__, 1).reshape(self.row, int(self.col / 3))
        result = np.sum(__input__, axis = 0)[2]
        return (result - self.post_cnt_lst[2]) * self.weight[4]


    def vid_cnt(self, __input__: list) -> float:
        """
        Parameters:
        ==========================
        - __input__: shape = (row * col, )
        """
        __input__ = np.expand_dims(__input__, 1).reshape(self.row, int(self.col / 3))
        result = np.sum(__input__, axis = 0)[3]
        return (result - self.post_cnt_lst[3]) * self.weight[5]


    def spec_kols(self, __input__: list) -> list:
        """
        Parameters:
        ===========================
        - __input__: shape = (row * col, )
        """
        if self.spec_kols_lst == ['None']:
            return 0
        else:
            __input__ = np.expand_dims(__input__, 1).reshape(self.row, int(self.col / 3))
            result = []
            for i in self.spec_kols_lst:
                result.append(np.sum(__input__, axis = 1)[int(i)])
            return [
                (result[i] - 1) * self.weight[6] for i in range(len(result))
            ]


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
            post_cnt_lst: list, spec_kols_lst: list, label_path: str,
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
            np.array(load_data(self.label_path)['platform'])[:self.row], axis = 1
        )
        self.__lambda__ = np.full((self.row * int(self.col / 3), 1), self.lamb)
        self.__input__ = np.array([0.1] * self.row * int(self.col / 3))


    def constraint_output(
            self, val: np.ndarray, trans_count: np.ndarray, input_count: np.ndarray
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
                semi_result = np.sum(input_count, axis = 1)[int(i)]
                result.append(semi_result)
            spec_kols = [
                (result[i] - 1) * self.weight[6] for i in range(len(result))
            ]
        result = (self.budget - val.T @ self.price_data) * self.weight[0] + \
            (trans_count.T @ self.voice_data - self.voice) * self.weight[1] + \
            (np.sum(input_count, axis = 0)[0] - self.post_cnt_lst[0]) * self.weight[2] + \
            (np.sum(input_count, axis = 0)[1] - self.post_cnt_lst[1]) * self.weight[3] + \
            (np.sum(input_count, axis = 0)[2] - self.post_cnt_lst[2]) * self.weight[4] + \
            (np.sum(input_count, axis = 0)[3] - self.post_cnt_lst[3]) * self.weight[5] + \
            np.sum(spec_kols)
        return result.item()


    def objective(self, __input__: list, lamb=None) -> float:
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


    def objective_l1(self, __input__: list, lamb: list) -> float:
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


    def objective_l2(self, __input__: list, lamb: list) -> float:
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


    def objective_enet(self, __input__: list, lamb: list) -> float:
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
            self.alpha * l1_term.item() + ((1 - self.alpha) / 2) * l2_term.item()


    def get_functions(self, types: str) -> tuple:
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


    def run_opt(self, func, cons_func: dict) -> tuple:
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
            fun = func,
            x0 = self.__input__,
            args = ([0] if func is self.objective else (self.__lambda__, )),
            method = 'SLSQP',
            bounds = [(0, 100)] * self.row * int(self.col / 3),
            constraints = cons_func,
            options = {'maxiter': 500, 'disp': True}
        )
        end_time = time.time()
        print("===================================================")
        print("End of optimization")
        print(f"Time elapsed: {end_time - start_time} seconds")
        return solution, solution.x


    def get_opt_result(self, solution) -> tuple:
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
            axis = 1
        )
        final_result = matrix_greed_search(final_result)
        # print(f"Adjusted Parameters matrix: \n {final_result}")
        result_with_label = np.concatenate((self.label_data, final_result), axis = 1)
        c_cost = count_cost(final_result, self.col)
        c_voice = count_voice(final_result, self.col)
        print(f"Output Cost: {c_cost} (Thousand dollars)")
        print(f"Output Voice: {c_voice}")
        return c_cost, c_voice, result_with_label


    def adjust_weight(
            self, obj_function, constraints: dict, budget: float, voice: float
        ) -> tuple:
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
        ## base case
        if (budget < self.budget and voice > self.voice) or (self.count > 200):
            solution, _ = self.run_opt(obj_function, constraints)
            print("Optimization terminated")
            _, _, result_with_label = self.get_opt_result(solution)
            print(to_df(result_with_label))
            return result_with_label

        ## recursion
        if (budget > self.budget and voice < self.voice) or \
            (budget < self.budget and voice < self.voice) or \
            (budget > self.budget and voice > self.voice):
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
                if self.weight[idx] > mean_weight:
                    self.weight[idx] = self.weight[idx] * 1.3 * multiplier
                if self.weight[idx] < mean_weight:
                    self.weight[idx] = self.weight[idx] * 0.77 * multiplier
            print(f"Iterate: {self.count}. Adjusted weight: {self.weight}")
            solution, _ = self.run_opt(obj_function, constraints)
            comp_cost, comp_voice, result_with_label = self.get_opt_result(solution)
            print(to_df(result_with_label))
            result_with_label = self.adjust_weight(
                obj_function, constraints, comp_cost, comp_voice
            )
            return result_with_label


    # def add_weight(
            # self, obj_function, constraints: dict, budget: float, voice: float
        # ) -> tuple:
        # """
        # Parameters
        # ----------
        #
        # - obj_function:
            # objective function we use
        # - constraints:
            # constraints we use
        # - remaining:
            # remaining constraints after adjusting
        # - budget:
            # current budget
        # - voice:
            # current voice
            #
        # obj_function, constraints: type <method>
        # """
        # base case
        # if (len(self.weight) == 1) or (budget < self.budget and voice > self.voice):
            # solution, solution_matrix = self.run_opt(obj_function, constraints)
            # print("Optimization terminated")
            # return solution, solution_matrix

        # recursion
        # if (budget > self.budget and voice < self.voice) or \
            # (budget < self.budget and voice < self.voice) or \
            # (budget > self.budget and voice > self.voice):
            # print("Constraint not satisfied, start adjusting constraint weight...\n")
            # obj_result = obj_function(self.__input__, self.__lambda__)
            # if constraints[-1]['fun'].__name__ == "constraint_voice":
                # cons_result = constraints[-1]['fun'](self.__input__, self.voice)
            # elif constraints[-1]['fun'].__name__ == "constraint_spec_kols":
                # cons_result = np.sum(constraints[-1]['fun'](self.__input__))
            # else:
                # cons_result = constraints[-1]['fun'](self.__input__)
            # print(f"Constraint to pop: {constraints[-1]['fun'].__name__}")
            # obj_result += cons_result * self.weight[-1]
            # self.weight.pop()
            # constraints.pop()
            # solution, _ = self.run_opt(obj_function, constraints)
            # comp_cost, comp_voice, _ = self.get_opt_result(solution)
            # solution, solution_matrix = self.add_weight(
                # obj_function, constraints, comp_cost, comp_voice
            # )
        # return solution, solution_matrix


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
