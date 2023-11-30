"""
Main execute function for optimization
"""
import os
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from .preproc import PreProc
from .opt_test_v3 import Optimizer
from .property_scoring import PropertyScoring


def arguments() -> ArgumentParser:
    """
    arguments parser function
    """
    parser = ArgumentParser()
    parser.add_argument(
        '--do_preproc', type=bool, default=False
    )
    parser.add_argument(
        '--dat_name', type=str, default="preproc_data_v4",
    )
    parser.add_argument(
        '--download',
        type = bool, default = False,
        help = 'Download data to local folder or not'
    )
    parser.add_argument(
        '-r', '--row', type = int, default = 5,
        help = 'Number of rows to be optimized'
    )
    parser.add_argument(
        '-t', '--type', type = str, default = 'plain',
        help = 'Type of objective function'
    )
    parser.add_argument(
        '-b', '--budget', type = int, default = 1000,
        help = 'Budget constraint'
    )
    parser.add_argument(
        '-v', '--voice', type = int, default = 5000,
        help = 'Target voice'
    )
    parser.add_argument(
        '-l', '--lamb', type = float, default = None,
        help = 'Lambda for regularization, and elastic net'
    )
    parser.add_argument(
        '-a', '--alpha', type = float, default = None,
        help = 'Alpha for elastic net'
    )
    parser.add_argument(
        '-p', '--post_cnt', type = int, nargs = '+', default = [0, 0, 0, 0],
        help = 'Post count constraint for each type, order: live, post, short, vid'
    )
    parser.add_argument(
        '-s', '--spec_kols', nargs = '+', default = [],
        help = 'Specific kols need to be selected'
    )
    parser.add_argument(
        '--w1', type=float, default=0.167,
        help="weight of promotion experience"
    )
    parser.add_argument(
        '--w2', type=float, default=0.833,
        help="weight of degree of robustness"
    )
    parser.add_argument(
        '--candidates', type=int, default=100,
        help="amount of candidates to be selected"
    )
    return parser.parse_args()


if __name__ == "__main__":
    ## ================= Parser ================= ##
    args = arguments()
    ## ================= Filter by scoring ================= ##
    __property__ = PropertyScoring(
        f'{os.getcwd()}/data/scoring_data_v3.csv', args.w1, args.w2
    )
    data_with_std = __property__.calculate_fod(__property__.scaled_data)
    org_data_full = __property__.main(data_with_std)
    org_data_full_filtered = org_data_full.iloc[:args.candidates, :]
    preproc_data = pd.read_csv(f'{os.getcwd()}/data/preproc_data_v3.csv')
    preproc_data = preproc_data[preproc_data['name'].isin(org_data_full_filtered['name'])]
    preproc_data.to_csv(f'{os.getcwd()}/data/preproc_data_v4.csv', index=False)
    ## ================= Preproc ================= ##
    if args.do_preproc:
        preprocessing = PreProc(args.dat_name, args.download)
        if args.dat_name.split("_")[0] == "preproc":
            preprocessing.for_preproc_data()
    else:
        pass
    ## ================= Optimizer ================= ##
    optimizer = Optimizer(
        f"{os.getcwd()}/data/use_data.csv", args.row, args.post_cnt, args.spec_kols,
        f"{os.getcwd()}/data/preproc_data_v4.csv", args.budget, args.voice, args.lamb,
        args.alpha
    )
    func, cons_func = optimizer.get_functions(args.type)
    solution, solution_matrix = optimizer.run_opt(func, cons_func)
    comp_cost, comp_voice, result_with_label = optimizer.get_opt_result(solution)
    result_with_label = optimizer.adjust_weight(
        func, cons_func, comp_cost, comp_voice
    )
    # for index, cons in enumerate(cons_func):
    #     cons['weight'] = optimizer.weight[index]
    # cons_func = sorted(cons_func, key = lambda x: x['weight'])
    optimizer.print_final_output(args.type)
    np.save(f'{os.getcwd()}/data/param_mat/all_{args.row}_{args.type}.npy', result_with_label)


