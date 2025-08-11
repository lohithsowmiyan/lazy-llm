import pandas as pd
import numpy as np
import random
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter
from dehb import DEHB
from sklearn.metrics import pairwise_distances_argmin_min
from src.utils.ezr import *


def dehbRuns(args,i_d):
    # ==== USER SETTINGS ==== #
            # budget

    target_cols = [nm.txt for nm in i_d.cols.y]
    #print(target_cols)
    # ==== LOAD DATASET ====
    
    df = pd.read_csv(args.dataset)
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    #print(df.info())
    chebyshevs = []
    for idx, r in df.iterrows():
        chebyshevs.append(chebyshev(i_d, r))
    #print(d2hs)
    #
    X = df.drop(columns=target_cols).values
    y = chebyshevs
    
   

    # ==== CONFIGURATION SPACE ====
    cs = ConfigurationSpace()
    for i, col in enumerate(df.drop(columns=target_cols).columns):
        col_vals = df[col].dropna().values  # drop NaN for min/max or unique
        # Integer columns
        if np.issubdtype(col_vals.dtype, np.integer):
            hp = UniformIntegerHyperparameter(f"x{i}", int(col_vals.min()), int(col_vals.max()))
        # Float columns
        elif np.issubdtype(col_vals.dtype, np.floating):
            hp = UniformFloatHyperparameter(f"x{i}", float(col_vals.min()), float(col_vals.max()))
        # String / object columns → categorical
        elif np.issubdtype(col_vals.dtype, np.object_) or isinstance(col_vals[0], str):
            unique_vals = sorted(set(col_vals))
            hp = CategoricalHyperparameter(f"x{i}", unique_vals)
        else:
            raise ValueError(f"Unsupported column type for {col}")
        
        cs.add_hyperparameter(hp)

    # ==== EVALUATION FUNCTION ====
    def evaluate_config(config, fidelity):
        """Map DEHB config to nearest row in dataset and return y value."""
        config_arr = np.array([config[f"x{i}"] for i in range(X.shape[1])]).reshape(1, -1)
        idx, _ = pairwise_distances_argmin_min(config_arr, X)
        return y[idx[0]]  # DEHB minimizes → use negative if y is to be maximized

    #print(cs.get_hyperparameters())
    # ==== RUN DEHB ====
    dehb = DEHB(
        f=evaluate_config,
        cs=cs,
        #dimensions=len(cs.get_hyperparameters()),
        min_fidelity=1, 
        max_fidelity=10,
        n_workers=1,
    )

    done = []
    for _ in range(args.last):
        job_info = dehb.ask()

        # Run the configuration for the given fidelity. Here you can freely distribute the computation to any worker you'd like.
        result = evaluate_config(config=job_info["config"], fidelity=job_info["fidelity"])
        done.append(result)
        result= {"fitness": result, "cost": 1}  # DEHB minimizes, so we return negative fitness
        # When you received the result, feed them back to the optimizer
        dehb.tell(job_info, result)

    print(done)


    # traj, runtime, history = dehb.run(fevals=args.last)

    # # ==== GET BEST ROW ====
    # best_idx = np.argmax(y) if y.max() == -min(traj) else np.argmin(y)  # depending on goal
    # best_row = df.iloc[best_idx]
    # print("Best row found:")
    # print(best_row)

    return sorted(done) #chebyshev(i_d,best_row)



# def run_dehb_on_rows(i, todo, done, b):
#     """
#     i     -> object with i.cols.x (column definitions) and d2h(i, row) for scoring
#     todo  -> list of candidate rows (from dataset)
#     done  -> list of already evaluated rows
#     b     -> budget (max number of evaluations)
#     """
    
#     # Create DEHB-friendly config space from dataset columns
#     from ConfigSpace import ConfigurationSpace
#     from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter

#     cs = ConfigurationSpace()
#     for col in i.cols.x:
#         if col.this.__name__ == 'NUM':
#             hp = UniformFloatHyperparameter(col.txt, col.lo, col.hi)
#         elif col.this.__name__ == 'SYM':
#             # Encode categorical as integer indices
#             hp = UniformIntegerHyperparameter(col.txt, 0, len(col.has) - 1)
#         cs.add_hyperparameter(hp)
    
#     # Dataset as numpy
#     dataset = np.array(todo + done, dtype=object)  # assuming todo+done = full dataset
    
#     # Mapping categorical values to integers for DEHB representation
#     sym_maps = {}
#     for col in i.cols.x:
#         if col.this.__name__ == 'SYM':
#             sym_maps[col.txt] = {val: idx for idx, val in enumerate(sorted(col.has))}
    
#     def row_to_numeric(row):
#         numeric_row = []
#         for col, val in zip(i.cols.x, row):
#             if col.this.__name__ == 'SYM':
#                 numeric_row.append(sym_maps[col.txt][val])
#             else:
#                 numeric_row.append(val)
#         return numeric_row
    
#     def numeric_to_row(num_row):
#         real_row = []
#         for col, val in zip(i.cols.x, num_row):
#             if col.this.__name__ == 'SYM':
#                 inv_map = {v: k for k, v in sym_maps[col.txt].items()}
#                 real_row.append(inv_map[int(round(val))])
#             else:
#                 real_row.append(val)
#         return real_row
    
#     numeric_dataset = np.array([row_to_numeric(row) for row in dataset], dtype=float)

#     # Evaluation function
#     def evaluate_config(config, budget):
#         config_arr = np.array([config[name] for name in cs.get_hyperparameter_names()]).reshape(1, -1)
#         # Find nearest row in dataset
#         from sklearn.metrics import pairwise_distances_argmin
#         idx = pairwise_distances_argmin(config_arr, numeric_dataset)[0]
#         candidate_row = dataset[idx]
        
#         # If candidate already in done, skip (no budget use)
#         if any(np.array_equal(candidate_row, r) for r in done):
#             return float("inf")  # large value → bad
        
#         done.append(candidate_row)
#         return -d2h(i, candidate_row)  # minimize → use -score if maximizing

#     dehb = DEHB(
#         f=evaluate_config,
#         cs=cs,
#         dimensions=len(cs.get_hyperparameters()),
#         min_budget=1,
#         max_budget=1
#     )

#     traj, runtime, history = dehb.run(fevals=b, verbose=True)
    
#     # Return best row from done
#     best_row = sorted(done, key=lambda r: d2h(i, r))[0]
#     return best_row

# def _ranked(i,lst:rows) -> rows:
#     "Sort `lst` by distance to heaven. Called by `_smo1()`."
#     lst = sorted(lst, key = lambda r:d2h(i,r))
#     return lst

def diffevols(args):
    

    i = DATA(csv(args.dataset))
    random.shuffle(i.rows)
    return dehbRuns(args, i)

    #return run_dehb_on_rows(i, i.rows[args.label:], _ranked(i,i.rows[:args.label]), args.last)

