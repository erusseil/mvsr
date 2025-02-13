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

    eggp_path = "/home/etru7215/Documents/mvsr_datasets/eggp/"
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
        self.numpy_expression = None
        self.expression = None
        self.model = None
        self.params = None

    def run(self):

        if not os.path.isfile(self.eggp_path + "/eggp"):
            raise ValueError("MvSR.eggp_path is incorrect, file not found. You must indicate the absolute path of the 'eggp' executable")
        
        command = f'{self.eggp_path}/eggp -d "{self.views_str}" -s {self.max_length} --nPop {self.pop_size} -g {self.generations} --non-terminals {self.operations} --number-params {self.n_params} --opt-retries {self.opt_retries} --moo --to-numpy --simplify'

        output_string = subprocess.check_output(command, shell=True, text=True)
        
        # Replace temporary the "[:" from the numpy expression to allow the split on the correct :
        output_string = output_string.replace('[:,', '$')
        output_table = pd.DataFrame([line.split(sep=',') for line in output_string[:-2].split(sep='\n')])

        # Replace back :
        output_table.iloc[1:, 1] = output_table.iloc[1:, 1].apply(lambda x: x.replace('$', '[:,'))

        output_table.columns = output_table.iloc[0]
        output_table.drop(index=0, inplace=True)

        self.raw_results = output_table
        self.numpy_expression = self.raw_results['Expression'].iloc[0]
        self.generate_visual_expression()

        model = MvSR.make_function(self.numpy_expression)
        self.model = model
        self.params = np.array([np.array((output_table['theta'].iloc[i]).split(sep=';')).astype(float) for i in range(len(output_table))])


    @staticmethod
    def make_function(expression):
        def func(x, t):
            return eval(expression)
        return func

    def generate_visual_expression(self):

        if type(self.raw_results)==type(None):
            print("Must run MvSR first")

        else:
            clean_expression = self.numpy_expression
            n_params_used = len(self.raw_results['theta'].iloc[0].split(sep=';'))

            for i in range(n_params_used):
                clean_expression = clean_expression.replace(f't[{i}]', alphabet[i])

            data_path = self.views_path[0]
            n_dim = np.shape(pd.read_csv(data_path))[1] - 1

            for i in range(n_dim):
                clean_expression = clean_expression.replace(f'x[:, {i}]', f'X{i}')

            self.expression = clean_expression
            print(self.expression)



    def plot_all_fits(self):

        for idx in range(len(self.raw_results)):
            data_path = self.views_path[idx]
            data = pd.read_csv(data_path)

            if np.shape(data)[1] != 2:
                print("Careful, the plot is adapted for 2D datasets. Only the first X axis will displayed.")
                
            X, Y = np.array(data.iloc[:, 0]), np.array(data.iloc[:, -1])
            parrays = self.params[idx]
        
            plt.figure()
            plt.scatter(X, Y)

            Xplot = np.linspace(X.min(), X.max(), 200)
            plt.plot(Xplot, self.model(np.array([Xplot]).T, parrays), label=MvSR.format_labels(parrays), color='red', alpha=0.8)
            plt.title(data_path)
            plt.legend()
        plt.show()

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
        