import pandas as pd
import numpy as np
from os.path import isfile, join
from os import listdir
import os
import shutil 
import string
import sys
import time

numpy_parser = {'exp':'np.exp', 'log':'np.log', 'abs':'np.abs',
               'sin':'np.sin', 'cos':'np.cos', 'tan':'np.tan',
               '^':'**', 'pow':"**", 'safe_log':'np.log', 'square':'np.square',
               "sqrt":"np.sqrt", "Exp":"np.exp", "Log":"np.log",
               "Pow":"**", "Square":"np.square", "Sqrt":"np.sqrt", "Abs":"np.abs"}

main_path = "/home/etru7215/Documents/MvSR/mvsr_datasets/datasets/"
datasets = {'linear':main_path + "linear/",
           'biology':main_path + "biology/",
           'galaxies':main_path + "galaxies/",
           'graphs':main_path + "log_graphs/",
           'supernovae':main_path + "supernovae/",
           'fluid':main_path + "fluid_mechanics/"}

seconds_timeout = 60 * 30

def make_function(expression):
    def func(x, t):
        return eval(expression)
    return func

def chi2(y_true, y_pred, n_freedom):

    # We normalize the values so that datasets with very different magnitude of values still produce comparable chi2
    norm = max(np.abs(y_true))
    norm_true, norm_pred = y_true/norm, y_pred/norm

    if n_freedom>0:
        return np.sum((norm_true - norm_pred) ** 2)/n_freedom

    else:
        return np.nan


def str_to_function(expr_str, params, x_symbols=['x0']):

    # Convert function names
    for element in numpy_parser:
        expr_str = expr_str.replace(element, numpy_parser[element])
    
    # Replace parameters
    for idx, p in enumerate(params):
        expr_str = expr_str.replace(p, f't[{idx}]')

    # Replace X axis
    for idx, x in enumerate(x_symbols):
        expr_str = expr_str.replace(x, f'x[:, {idx}]')

    return expr_str, make_function(expr_str)

def find_csv_filenames(path_to_dir):
    suffix=".csv"
    filenames = listdir(path_to_dir)
    files = [filename for filename in filenames if filename.endswith(suffix)]
    files.sort()
    return files

        
class general_MvSR():
    def __init__(self, algo, data_path, config, train_points, test_points, seed=0):
        if not algo in [MvSR_eggp, MvSR_operon, MvSR_PySR, MvSR_PhySO]:
            message = "The algorithm indicated is invalid. Should be: eggp, PySR, operon or PhySO"
            raise ValueError(message)

        self.data_path = data_path
        self.config = config
        self.seed = seed
        self.main = algo(data_path, config, train_points, test_points, seed)
        self.best_expression = None
        self.best_np_expression = None
        self.best_parameters = None
        self.best_model = None

    def run(self):
        self.best_expression, self.best_np_expression, self.best_parameters, self.best_model = self.main.run()


class MvSR_eggp():

    default_pop_size = 100
    default_generation = 100
    default_opt_retries = 3
    default_operators = 'add,sub,mul,div,exp,log,sqrt,abs,power,square'

    def __init__(self, data_path, config, train_points, test_points, seed=0):

        self.data_path = data_path
        self.max_size = config['max_size']
        self.max_params = config['max_params']
        self.train_points = train_points
        self.test_points = test_points
        self.to_input = self.from_path_to_input()
        self.seed = seed

    def from_path_to_input(self):
     
        onlyfiles = find_csv_filenames(self.data_path)
        X, y = [], []

        for idx, file in enumerate(onlyfiles):
            table = pd.read_csv(self.data_path + file)
            Xadd = table.iloc[self.train_points[idx], :-1].astype('float')
            X.append(np.array(Xadd))
            y.append(np.array(table.iloc[self.train_points[idx], -1]).astype('float'))

        return X, y
        
    def run(self):
        
        import eggp

        X, y = self.to_input
        MvSR = eggp.EGGP(maxSize=self.max_size, nPop=self.default_pop_size, nonterminals=MvSR_eggp.default_operators,
                       gen=self.default_generation, nParams=self.max_params, optIter=self.default_opt_retries)

    
        MvSR.fit_mvsr(X, y)
        pareto = MvSR.results
        
        best_solution = pareto.iloc[np.argmin(pareto['maxloss'])]['id']
        model, clean_expression = MvSR.get_model(best_solution)
        best_pareto = pareto[pareto['id']==best_solution]

        numpy_expression = best_pareto['Numpy'].iloc[0]
        params = np.array([np.array((best_pareto['theta'].iloc[i]).split(sep=';')).astype(float) for i in range(len(best_pareto))])
        
        return clean_expression, numpy_expression, params, model


class MvSR_operon():
    
    default_maxD = 10
    default_pop_size = 100
    default_generation = 100
    default_opt_retries = 3

    # Is defined in the __init__ because it requires an import
    default_operators = None

    def __init__(self, data_path, config, train_points, test_points, seed=0):

        import pyoperon as Operon
        # Always includes the 4 basic operators:
        MvSR_operon.default_operators = Operon.NodeType.Exp|Operon.NodeType.Log|Operon.NodeType.Pow|Operon.NodeType.Square|Operon.NodeType.Sqrt|Operon.NodeType.Abs

        self.data_path = data_path

        # Because parameters are not included in the size of operon equation we only take two thirds of the max size.
        # It not a perfect solution but it is often a good rule of thumb
        
        self.max_size = int(2/3 * config['max_size'])
        self.max_params = config['max_params']
        self.train_points = train_points
        self.test_points = test_points
        self.temp_path = temp_train_files(data_path, self.train_points)
        self.to_input = self.temp_path
        self.seed = seed


    def run(self):

        import operon_tools as pyop
        import operon_mvsr as mvsr

        files = find_csv_filenames(self.data_path)
        dimX = np.shape(pd.read_csv(self.data_path+files[0]))[1]-1
        
        agg_best_str, all_best_str = mvsr.MultiViewSR(self.data_path, maxL=self.max_size, maxD=self.default_maxD,
                                                      generations=self.default_generation, pop_size=self.default_pop_size,
                                                      opt_retries=self.default_opt_retries, seed=self.seed,
                                                      OperationSet=self.default_operators, verbose=False, explicit_params=False)

        # This first conversion replaces floats with parameters.
        func, func_str, initial_guess = pyop.convert_string_to_func(agg_best_str, 1)
        param_names = list(string.ascii_uppercase)[:len(initial_guess)]
        
        # The second conversion ensures that the same format is applied to all MvSR methods. Not necessary per say.
        np_str, std_func = str_to_function(func_str, param_names, x_symbols=[f"X{k+1}" for k in range(dimX)])

        if initial_guess != {}:
            params = self.reoptimize_parameters(func, self.data_path, initial_guess)
        else:
            params = []
        shutil.rmtree(self.temp_path)

        return func_str, np_str, params, std_func

    @staticmethod
    def reoptimize_parameters(func, path, initial_guess):

        import operon_tools as pyop
        import operon_mvsr as mvsr

        onlyfiles = find_csv_filenames(path)

        all_params = []
        for file in onlyfiles:
            _, params = pyop.refit_solution(func, path+file, initial_guess)
            all_params.append(params)

        return np.array(all_params)
            

class MvSR_PySR():

    default_pop_size = 100
    default_generation = 100
    default_opt_retries = 3
    default_operators_unary = ["exp", "abs", "square", "sqrt", "log"]
    default_operators_binary = ["+", "*", "-", "/", "^"]


    def __init__(self, data_path, config, train_points, test_points, seed=0):

        self.data_path = data_path
        self.max_size = config['max_size']
        self.max_params = config['max_params']
        self.train_points = train_points
        self.test_points = test_points
        self.to_input = self.from_path_to_input()
        self.seed = seed


    def from_path_to_input(self):
     
        onlyfiles = find_csv_filenames(self.data_path)
    
        X, y, category = [], [], []

        for idx, file in enumerate(onlyfiles):
            table = pd.read_csv(self.data_path + file)
            Xadd = table.iloc[self.train_points[idx], :-1]
            X += [arr.tolist() for arr in list(Xadd.values)]
            y += list(table.iloc[self.train_points[idx], -1].values.flatten())
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
            timeout_in_seconds=seconds_timeout,
            random_state=self.seed,
            deterministic=True,
            parallelism='serial',
            verbosity=0,
        )
        print('fitting ...')
        model.fit(X, y, category=category)

        best = model.get_best()
        parameters = np.array(best['julia_expression'].metadata.parameters).T
        param_names = sorted(set(re.findall(r'p\d+', best.equation)))

        np_str, func = str_to_function(best.equation, param_names)
        
        return best.equation, np_str, parameters, func


class MvSR_PhySO():

    default_generation = 100
    default_opt_retries = 3
    default_operators = ["add", "sub", "mul", "div", "log", "exp", "pow", 'abs', "n2", "sqrt"]
    default_batch_size = 100
    default_max_n_evaluations = 10000


    def __init__(self, data_path, config, train_points, test_points, seed=0):

        self.data_path = data_path
        self.max_size = config['max_size']
        self.max_params = config['max_params']
        self.train_points = train_points
        self.test_points = test_points
        self.to_input = self.from_path_to_input()
        self.seed = seed

    def from_path_to_input(self):
     
        onlyfiles = find_csv_filenames(self.data_path)
        X, y = [], []

        for idx, file in enumerate(onlyfiles):
            table = pd.read_csv(self.data_path + file)
            Xadd = table.iloc[self.train_points[idx], :-1].astype('float')
            X.append(np.array(Xadd).T)
            y.append(np.array(table.iloc[self.train_points[idx], -1]).astype('float'))

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

        config = physo.config.config_mvsr0_test.config_mvsr0_test
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

def temp_train_files(data_path, train_points):
    
    trimmed = data_path.rstrip('/\\')
    last_pos = len(trimmed) - 1 if trimmed else -1
    temp_path = data_path[:last_pos+1] + "_temp/"
    
    if os.path.isdir(temp_path):
        shutil.rmtree(temp_path)
    
    os.mkdir(temp_path)
    
    onlyfiles = find_csv_filenames(data_path)
    for i in range(len(onlyfiles)):
        data = pd.read_csv(data_path+onlyfiles[i])
        temp_data = data.iloc[train_points[i]]
        temp_data.to_csv(temp_path+onlyfiles[i], index=False)

    return temp_path
        
def compute_score(path, func, parameters, train_points, test_points):

        onlyfiles = find_csv_filenames(path)
        train_metrics = []
        test_metrics = []
    
        for idx, file in enumerate(onlyfiles):
            table = pd.read_csv(path + file)
            X_train = np.array(table.iloc[train_points[idx], :-1])
            y_train = np.array(table.iloc[train_points[idx], -1])
            y_pred_train = func(X_train, parameters[idx])
            n_freedom_train = len(X_train) - len(parameters[idx])
            train_metrics.append(float(chi2(y_train, y_pred_train, n_freedom_train)))
            
            X_test = np.array(table.iloc[test_points[idx], :-1])
            y_test = np.array(table.iloc[test_points[idx], -1])
            y_pred_test = func(X_test, parameters[idx])
            n_freedom_test = len(X_test) - 0 # We are not refitting, so no parameters are ajusted
            test_metrics.append(float(chi2(y_test, y_pred_test, n_freedom_test)))

        return train_metrics, test_metrics


if __name__ == "__main__":
    
    # To run an analysis: 'python main.py dataset method'
    
    dataset = sys.argv[1]
    method = sys.argv[2]
    
    n_run = 10
    test_percent = 0.2
    
    methods = {'PySR':MvSR_PySR, 'operon':MvSR_operon,
               'PhySO':MvSR_PhySO, 'eggp':MvSR_eggp}
    
    configs = {'small_simple':{'max_size':15, 'max_params':2},
               'small_complex':{'max_size':15, 'max_params':4},
               'big_simple':{'max_size':30, 'max_params':2},
               'big_complex':{'max_size':30, 'max_params':4}}
    

    for idx2, config in enumerate(configs):
    
        numpy_expression = []
        parameters = []
        scores_train = []
        scores_test = []
        train_data = []
        computation_time = []
    
        for idx in range(n_run):
            print(dataset, config, idx)
            start = time.time()
            
            np.random.seed(seed=idx)

            lengths = [len(pd.read_csv(datasets[dataset]+k)) for k in find_csv_filenames(datasets[dataset])]
            
            test_points = [np.random.choice(k, max(1, int(test_percent*k))) for k in lengths]
            train_points = [np.setdiff1d(np.arange(lengths[k]), test_points[k]) for k in range(len(test_points))]

            analysis = general_MvSR(methods[method], datasets[dataset], configs[config], train_points, test_points, seed=idx)
            analysis.run()

            score_train, score_test = compute_score(analysis.data_path, analysis.best_model, analysis.best_parameters, train_points, test_points)
            numpy_expression += [analysis.best_np_expression]
            parameters += [analysis.best_parameters]
            scores_train += [score_train]
            scores_test += [score_test]
            computation_time += [time.time()-start]
    
        df = pd.DataFrame(data={'numpy_expression':numpy_expression,
                                'parameters':parameters, 'chi2':scores_train, 'chi2_test':scores_test,
                                'computation_time':computation_time})

        file_name = f'results/{method}/{dataset}_{config}.pkl'
        df.to_pickle(file_name)
    
        print(file_name, " OK")