import numpy as np
import random
from scipy.optimize import fmin_l_bfgs_b
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from src.utils.ezr import *

import warnings

# Suppress specific warning
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Predicted variances smaller than 0. Setting those variances to 0.")

def _UCB_GPM(i, todo, done):
    kernel = C(1.0, (1e-8, 1e8)) * RBF(1.0, (1e-8, 1e8))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, optimizer=None)
    
    num_indexes = [col.at for col in i.cols.x if col.this.__name__ == 'NUM']
    sym_indexes = [col.at for col in i.cols.x if col.this.__name__ == 'SYM']
    
    num_transformer = StandardScaler()
    cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_indexes),
            ('cat', cat_transformer, sym_indexes)])
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    if sym_indexes:
        cat_data = np.array([[str(row[idx]) for idx in sym_indexes] for row in done], dtype=object)
        cat_transformer.fit(cat_data)
    
    def custom_optimizer(obj_func, initial_theta, bounds):
        theta_opt, func_min, _ = fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds, maxiter=1000)
        return theta_opt, func_min
    
    gp.optimizer = custom_optimizer
    
    def update_gp_model(done_set):
        X_done = np.array([x for x in done_set], dtype=object)
        y_done = np.array([-d2h(i, x) for x in done_set])
        X_done_transformed = pipeline.fit_transform(X_done)
        gp.fit(X_done_transformed, y_done)
    
    def ucb(x, kappa=2.576):
        x = np.array(x).reshape(1, -1).astype(object)
        x_transformed = pipeline.transform(x)
        mean, std = gp.predict(x_transformed, return_std=True)
        return mean + kappa * std
    
    while todo and len(done) < the.Last:
        update_gp_model(done)
        random.shuffle(todo)
        todo_subset = todo[:the.any]

        ucb_values = [ucb(row) for row in todo_subset]
        best_idx = np.argmax(ucb_values)
        best_candidate = todo.pop(best_idx)
         
        done.append(best_candidate)
    
    return _ranked(done)

def _ranked(i,lst:rows) -> rows:
    "Sort `lst` by distance to heaven. Called by `_smo1()`."
    lst = sorted(lst, key = lambda r:d2h(i,r))
    return lst

def ucbs(args):
    i = DATA(csv(args.dataset))
    return _UCB_GPM(i, i.rows[args.label:], _ranked(i,i.rows[:args.label]))