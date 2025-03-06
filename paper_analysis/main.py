import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from os import listdir
import string
import sys
import time

numpy_parser = {'exp':'np.exp', 'log':'np.log', 'abs':'np.abs',
               'sin':'np.sin', 'cos':'np.cos', 'tan':'np.tan',
               '^':'**', 'pow':"**", 'safe_log':'np.log', 'square':'np.square',
               "sqrt":"np.sqrt"}

datasets = {'linear':"/home/etru7215/Documents/mvsr_datasets/datasets/linear/",
           'biology':"/home/etru7215/Documents/mvsr_datasets/datasets/biology/",
           'galaxies':"/home/etru7215/Documents/mvsr_datasets/datasets/galaxies/",
           'graphs':"/home/etru7215/Documents/mvsr_datasets/datasets/log_graphs/",
           'supernovae':"/home/etru7215/Documents/mvsr_datasets/datasets/supernovae/",
           'fluid':"/home/etru7215/Documents/mvsr_datasets/datasets/fluid_mechanics/"}


def make_function(expression):
    def func(x, t):
        return eval(expression)
    return func

def chi2(y_true, y_pred, n_params):

    # We normalize the values so that datasets with very different magnitude of values still produce comparable chi2
    norm = max(np.abs(y_true))
    norm_true, norm_pred = y_true/norm, y_pred/norm
    
    return np.sum((norm_true - norm_pred) ** 2)/(len(y_true) - n_params)


def str_to_function(expr_str, params, x_symbols=['x0']):

    # First convert it to a numpy str
    # Replace parameters
    for idx, p in enumerate(params):
        expr_str = expr_str.replace(p, f't[{idx}]')

    # Replace X axis
    for idx, x in enumerate(x_symbols):
        expr_str = expr_str.replace(x, f'x[:, {idx}]')

    # Convert function names
    for element in numpy_parser:
        expr_str = expr_str.replace(element, numpy_parser[element])
    
    return expr_str, make_function(expr_str)

def find_csv_filenames(path_to_dir):
    suffix=".csv"
    filenames = listdir(path_to_dir)
    files = [filename for filename in filenames if filename.endswith(suffix)]
    files.sort()
    return files

        
class general_MvSR():
    def __init__(self, algo, data_path, config, seed=0):
        if not algo in [MvSR_eggp, MvSR_pyoperon, MvSR_PySR, MvSR_PhySO]:
            message = "The algorithm indicated is invalid. Should be: eggp, PySR, pyoperon or PhySO"
            raise ValueError(message)

        self.data_path = data_path
        self.config = config
        self.seed = seed
        self.main = algo(data_path, config, seed)
        self.best_expression = None
        self.best_np_expression = None
        self.best_parameters = None
        self.best_model = None

    def run(self):
        self.best_expression, self.best_np_expression, self.best_parameters, self.best_model = self.main.run()


class MvSR_eggp():

    default_pop_size = 100
    default_generation = 300
    default_opt_retries = 3
    default_operators = 'add,sub,mul,div,exp,log,sqrt,abs,pow,square'

    def __init__(self, data_path, config, seed=0):

        self.data_path = data_path
        self.max_size = config['max_size']
        self.max_params = config['max_params']
        self.to_input = data_path
        self.seed = seed

    def run(self):


        import sys
        import os
        sys.path.append(os.path.abspath("/home/etru7215/Documents/mvsr_datasets/eggp/"))
        import multiview as mv


        MvSR = mv.MvSR(self.data_path, max_length=self.max_size, pop_size=self.default_pop_size,
                       generations=self.default_generation, n_params=self.max_params, opt_retries=self.default_opt_retries)

        MvSR.run()
        return MvSR.expression, MvSR.numpy_expression, MvSR.params, MvSR.model


class MvSR_pyoperon():
    
    default_maxD = 10
    default_pop_size = 1000
    default_generation = 2000
    default_opt_retries = 10

    # Is defined in the __init__ because it requires an import
    default_operators = None

    def __init__(self, data_path, config, seed=0):

        import pyoperon as Operon
        # Always includes the 4 basic operators:
        MvSR_pyoperon.default_operators = Operon.NodeType.Exp|Operon.NodeType.Log|Operon.NodeType.Pow|Operon.NodeType.Square|Operon.NodeType.Sqrt|Operon.NodeType.Abs

        self.data_path = data_path

        # Because parameters are not included in the size of pyoperon equation we only take two thirds of the max size.
        # It not a perfect solution but it is often a good rule of thumb
        
        self.max_size = int(2/3 * config['max_size'])
        self.max_params = config['max_params']
        self.to_input = data_path
        self.seed = seed


    def run(self):

        import sys
        import os
        sys.path.append(os.path.abspath("/home/etru7215/Documents/mvsr_datasets/pyoperon/"))
        import analysis as pyop
        import mvsr as mvsr

        files = find_csv_filenames(self.data_path)
        dimX = np.shape(pd.read_csv(self.data_path+files[0]))[1]-1
        
        agg_best_str, all_best_str = mvsr.MultiViewSR(self.data_path, maxL=self.max_size, maxD=self.default_maxD,
                                                      generations=self.default_generation, pop_size=self.default_pop_size,
                                                      opt_retries=self.default_opt_retries, seed=self.seed,
                                                      verbose=False, explicit_params=False)

        # This first conversion replaces floats with parameters.
        func, func_str, initial_guess = pyop.convert_string_to_func(agg_best_str, 1)
        param_names = list(string.ascii_uppercase)[:len(initial_guess)]
        
        # The second conversion ensures that the same format is applied to all MvSR methods. Not necessary per say.
        np_str, std_func = str_to_function(func_str, param_names, x_symbols=[f"X{k+1}" for k in range(dimX)])
        
        params = self.reoptimize_parameters(func, self.data_path, initial_guess)

        return func_str, np_str, params, std_func

    @staticmethod
    def reoptimize_parameters(func, path, initial_guess):

        import sys
        import os
        sys.path.append(os.path.abspath("/home/etru7215/Documents/mvsr_datasets/pyoperon/"))
        import analysis as pyop
        import mvsr as mvsr

        onlyfiles = find_csv_filenames(path)

        all_params = []
        for file in onlyfiles:
            _, params = pyop.refit_solution(func, path+file, initial_guess)
            all_params.append(params)

        return np.array(all_params)
            

class MvSR_PySR():

    default_pop_size = 100
    default_generation = 100
    default_opt_retries = 2
    default_operators_unary = ["exp", "abs", "square", "sqrt", "log"]
    default_operators_binary = ["+", "*", "-", "/", "^"]


    def __init__(self, data_path, config, seed=0):

        self.data_path = data_path
        self.max_size = config['max_size']
        self.max_params = config['max_params']
        self.to_input = self.from_path_to_input()
        self.seed = seed


    def from_path_to_input(self):
     
        onlyfiles = find_csv_filenames(self.data_path)
    
        X, y, category = [], [], []

        for idx, file in enumerate(onlyfiles):
            table = pd.read_csv(self.data_path + file)
            Xadd = table.iloc[:, :-1]
            X += [arr.tolist() for arr in list(Xadd.values)]
            y += list(table.iloc[:, -1].values.flatten())
            category += [idx]*len(Xadd)

        return np.array(X), np.array(y), np.array(category) 


    def run(self):

        from pysr import PySRRegressor, ParametricExpressionSpec
        import warnings
        warnings.simplefilter('ignore', UserWarning)
        import re


        X, y, category = self.to_input

        expression_spec = ParametricExpressionSpec(max_parameters=self.max_params)
        model = PySRRegressor(
            expression_spec=expression_spec,
            maxsize=self.max_size,
            population_size=self.default_pop_size,
            niterations=self.default_generation,
            optimizer_nrestarts=self.default_opt_retries,
            binary_operators=self.default_operators_binary,
            unary_operators=self.default_operators_unary,
            random_state=self.seed,
            deterministic=True,
            parallelism='serial',
            verbosity=0,
        )
        model.fit(X, y, category=category)

        best = model.get_best()
        parameters = np.array(best['julia_expression'].metadata.parameters).T
        param_names = sorted(set(re.findall(r'p\d+', best.equation)))

        np_str, func = str_to_function(best.equation, param_names)
        
        return best.equation, np_str, parameters, func


class MvSR_PhySO():

    default_generation = 10
    default_opt_retries = 3
    default_operators = ["add", "sub", "mul", "div", "log", "exp", "pow", 'abs', "n2", "sqrt"]
    default_batch_size = 100
    default_max_n_evaluations = 10000


    def __init__(self, data_path, config, seed=0):

        self.data_path = data_path
        self.max_size = config['max_size']
        self.max_params = config['max_params']
        self.to_input = self.from_path_to_input()
        self.seed = seed

    def from_path_to_input(self):
     
        onlyfiles = find_csv_filenames(self.data_path)
        X, y = [], []

        for idx, file in enumerate(onlyfiles):
            table = pd.read_csv(self.data_path + file)
            Xadd = table.iloc[:, :-1]
            X.append(np.array(Xadd).T)
            y.append(np.array(table.iloc[:, -1]))

        return X, y

    def run(self):

        import warnings
        warnings.simplefilter('ignore', UserWarning)

        import torch
        import physo
        import physo.learn.monitoring as monitoring
        import re

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        files = find_csv_filenames(self.data_path)
        dimX = np.shape(pd.read_csv(self.data_path+files[0]))[1]-1
        X_names = [f"x{k}" for k in range(dimX)]
        
        multi_X, multi_y = self.to_input

        run_logger     = lambda : monitoring.RunLogger()
        run_visualiser = lambda : monitoring.RunVisualiser(do_show   = False,
                                                   do_prints = False,
                                                   draw_all_progs_fit=False)

        config = physo.config.config_mvsr0.config_mvsr0
        config['learning_config']['max_time_step'] = self.max_size
        config['learning_config']['batch_size'] = self.default_batch_size
        config['free_const_opti_args']['method_args']['lbfgs_func_args']['max_iter'] = self.default_opt_retries
        config['priors_config'][0][1]['max_length'] = self.max_size
        
        # Running SR task
        expression, logs = physo.ClassSR(multi_X, multi_y,
                                    X_names = X_names,
                                    y_name  = "y",
                                    spe_free_consts_names = [f"p{i}" for i in range(self.max_params)],
                                    op_names = self.default_operators,
                                    get_run_logger     = run_logger,
                                    get_run_visualiser = run_visualiser,
                                    class_free_consts_names = ["k0", "k1"],
                                    parallel_mode = False,
                                    run_config = config,
                                    epochs = self.default_generation,
                                    max_n_evaluations = self.default_max_n_evaluations)

        str_expr = expression.get_infix_str()
        param_names = sorted(set(re.findall(r'p\d+', str_expr)))
        const_values = np.array(expression.free_consts.class_values[0])

        for idx, k in enumerate(["k0", "k1"]):
            str_expr = str_expr.replace(k, str(const_values[idx]))
        
        np_str, func = str_to_function(str_expr, param_names, x_symbols=X_names)

        all_params = np.array(expression.free_consts.spe_values)[0].T
        used_params = all_params[:, [int(k[1:]) for k in param_names]]

        return str_expr, np_str, used_params, func


def compute_score(path, func, parameters):

        onlyfiles = find_csv_filenames(path)
        metrics = []
        for idx, file in enumerate(onlyfiles):
            table = pd.read_csv(path + file)
            X = np.array(table.iloc[:, :-1])
            y = np.array(table.iloc[:, -1])
            y_pred = func(X, parameters[idx])
            metrics.append(float(chi2(y, y_pred, len(parameters[idx]))))

        return metrics


if __name__ == "__main__":
    
    # To run an analysis: 'python main.py dataset method'
    
    dataset = sys.argv[1] # linear
    method = sys.argv[2] # eggp
    
    n_run = 10
    
    methods = {'PySR':MvSR_PySR, 'pyoperon':MvSR_pyoperon,
               'PhySO':MvSR_PhySO, 'eggp':MvSR_eggp}
    
    configs = {'small_simple':{'max_size':15, 'max_params':2},
               'small_complex':{'max_size':15, 'max_params':4},
               'big_simple':{'max_size':30, 'max_params':2},
               'big_complex':{'max_size':30, 'max_params':4}}
    
    
    for idx2, config in enumerate(configs):
    
        numpy_expression = []
        parameters = []
        scores = []
        computation_time = []
    
        for idx in range(n_run):
            start = time.time()
            analysis = general_MvSR(methods[method], datasets[dataset], configs[config], seed=idx)
            analysis.run()
    
            score = compute_score(analysis.data_path, analysis.best_model, analysis.best_parameters)
            numpy_expression += [analysis.best_np_expression]
            parameters += [analysis.best_parameters]
            scores += [score]
            computation_time += [time.time()-start]
    
        df = pd.DataFrame(data={'numpy_expression':numpy_expression,
                                'parameters':parameters, 'chi2':scores,
                                'computation_time':computation_time})
        
        file_name = f'results/{method}/{dataset}_{config}.pkl'
        df.to_pickle(file_name)
    
        print(file_name, " OK")