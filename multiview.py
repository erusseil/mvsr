import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp
import subprocess
import string
from os import listdir
import glob, os
from os.path import isfile, join


alphabet = list(string.ascii_uppercase)


class MvSR():
    
    """Initiate a MvSR analysis linked to a folder of views. Default values are suited for simple problem and should be tuned to the user specific challenges.

    Parameters
    ----------

    data_path: str
        Path to the folder containing the views (in .csv)
    max_length: int
        Maximum size of the equation to generate. 
        Optional. Default is 20.
    n_params: int
        Maximum number of parameters of the equation to generate.
        Optional. Default is 5.
    generations: int
        Number of generation for the genetic symbolic regression evolution.
        Optional. Default is 200.
    pop_size: int
        Number of individual per generation for the genetic symbolic regression evolution.
        Optional. Default is 100.
    opt_retries: int
        Number of run of the optimizer on each example for each generated function. Increasing this number may largely improve the quality of the results for certain equations, such as equations based on exp, for which minimization is more challenging. However it will result in an increase of the computation time. 
        Optional. Default is 5.
    operations: str
        Operators allowed for the equation to generate.
        Optional. Default is 'add,sub,mul,div,exp,log,sqrt,abs'
    """

    default_max_length = 20
    default_n_params = 5
    default_generations = 200
    default_pop_size = 100
    default_opt_retries = 5
    default_operations = 'add,sub,mul,div,exp,log,sqrt,abs'

    def __init__(self, data_path, max_length=None, n_params=None, generations=None,
                 pop_size=None, opt_retries=None, operations=None):

        self.data_path = data_path
        self.views_path = []
        
        for file in os.listdir(data_path):
            if file.endswith(".csv"):
                self.views_path += [os.path.join(data_path, file)]

        self.views_str = ''
        for view in self.views_path:
            self.views_str += f"{view} "

        self.max_length = MvSR.default_max_length if max_length is None else max_length
        self.pop_size = MvSR.default_pop_size if pop_size is None else pop_size
        self.generations = MvSR.default_generations if generations is None else generations
        self.n_params = MvSR.default_n_params if n_params is None else n_params
        self.opt_retries = MvSR.default_opt_retries if opt_retries is None else opt_retries
        self.operations = MvSR.default_operations if operations is None else operations

        self.raw_results = None
        self.expression = None
        self.model = None

    def run(self):
        command = f'./eggp -d "{self.views_str}" -s {self.max_length} --nPop {self.pop_size} -g {self.generations} --non-terminals {self.operations} --number-params {self.n_params} --distribution MSE --opt-retries {self.opt_retries} --moo'

        output_string = subprocess.check_output(command, shell=True, text=True)
        output_table = pd.DataFrame([line.split(sep=',') for line in output_string[:-2].split(sep='\n')])
        output_table.columns = output_table.iloc[0]
        output_table.drop(index=0, inplace=True)
        self.raw_results = output_table
        self.convert_raw_results_to_expression()
        self.convert_expression_to_function()
        print(self.expression)

    def convert_raw_results_to_expression(self):

        if type(self.raw_results)==type(None):
            print("Must run MvSR first")

        else:
            all_expressions = []
            for idx in range(len(self.raw_results)):
                expression = self.raw_results['Expression'].iloc[idx]
                for idx2, theta in enumerate(self.raw_results['theta'].iloc[idx].split(sep=";")):
                    if theta != '':
                        expression = expression.replace(theta, alphabet[idx2])
        
                all_expressions += [expression]
            
            if len(set(all_expressions)) == 1:
                self.expression = sp.simplify(all_expressions[0])
            
            
            else:
                print("Careful, parsing probably went wrong")
                print(all_expressions)

    def convert_expression_to_function(self):

        if type(self.expression) == type(None):
            print("Must convert_raw_results_to_expression first")

        else:
            used_params = alphabet[:MvSR.check_n_param_used(self.expression)]
            func = sp.lambdify(
                        ["x0"] + used_params,
                        str(self.expression),
                        modules=[{"Exp": np.exp, "Log": np.log, "Sin": np.sin, "Abs": np.abs, "Sqrt": np.sqrt}, "numpy"],
                    )

            self.model = func

    def plot_all_fits(self):
    
        for idx in range(len(self.raw_results)):
            params = self.raw_results['theta'].iloc[idx]
            data_path = self.views_path[idx]
            data = pd.read_csv(data_path)
            X, Y = np.array(data.iloc[:, 0]), np.array(data.iloc[:, -1])
            parrays = np.array(params.split(sep=';')).astype(float)
        
            plt.figure()
            plt.scatter(X, Y)
            
            Xplot = np.linspace(X.min(), X.max(), 200)
            plt.plot(Xplot, self.model(Xplot, *parrays), label=MvSR.format_labels(parrays), color='red', alpha=0.8)
            plt.title(data_path)
            plt.legend()

    @staticmethod
    def format_labels(params, rounding=3):
        
        label = ''
        for i in range(len(params)):
            value = params[i]
            name = alphabet[i]
            label += f"{name}={round(value, rounding)}"

            if i!=len(params)-1:
                label += "\n"

        return label
        

    @staticmethod
    def check_n_param_used(expression):

        n_used = 0
        loop = True
        while loop:
            if alphabet[n_used] in str(expression):
                n_used += 1
        
            else:
                loop=False

        return n_used
        