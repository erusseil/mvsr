import pandas as pd
import numpy as np
import main
import pickle
import matplotlib.pyplot as plt
import sympy as sp
from IPython.display import display, Math


configs = ['small_simple', 'small_complex', 'big_simple', 'big_complex']
methods = ['PySR', 'eggp', 'PhySO', 'operon']
datasets = ['supernovae', 'galaxies', 'graphs', 'fluid', 'biology']


def summary_tables(gather, path="results/"):

    results = {}
    
    for dataset in datasets:

        result = pd.DataFrame(columns=methods, index=configs)
        
        for config in configs:
            for method in methods:
                
                df = pd.read_pickle(f'{path}/{method}/{dataset}_{config}.pkl')

                # Remove the abs after PhySo is fixed
                best_run = df.iloc[np.argmin(df['chi2'].apply(max).apply(fix_chi2))]
                
                if gather == 'chi2':
                    result.loc[config, method] = max(best_run['chi2'])

                elif gather == 'chi2_test':
                    result.loc[config, method] = max(best_run['chi2_test'])

                elif gather == 'time':
                    result.loc[config, method] = df['computation_time'].mean()

                elif gather == 'parameters':
                    result.loc[config, method] = best_run['parameters']

                elif gather == 'expression':
                    result.loc[config, method] = best_run['numpy_expression']

                else:
                    print("Choose between chi2, chi2_test, time, expression, or parameters")

        results[dataset] = result

    return results


def overall_best(chi2_table):
    min_location = chi2_table.stack().idxmin()  # Get the (index, column) locatio
    return min_location[0], min_location[1] #Row, Column

def fix_chi2(chi2_list):
    chi2_arr = np.array(chi2_list)
    positive = np.where(chi2_arr>=0, chi2_arr, 999)
    return np.min(positive)
    
def plot_solution(ax, dataset, model, parameters, show_n=10, title='', scatter_alpha=0.8, xlim=None):

    onlyfiles = main.find_csv_filenames(main.datasets[dataset])
    
    # Setup X linspace
    all_X = [pd.read_csv(main.datasets[dataset] + k).iloc[:, 0].values for k in onlyfiles[:show_n]]
    minX, maxX = max([max(n) for n in all_X]), min([min(n) for n in all_X])
    X_plot = np.array([np.linspace(minX, maxX, 3000)]).T
    
    all_y = [pd.read_csv(main.datasets[dataset] + k).iloc[:, -1].values for k in onlyfiles[:show_n]]
    maxy, miny = max([max(n) for n in all_y]), min([min(n) for n in all_y])
    margin = 0.1 * (maxy-miny)
        
    for idx, file in enumerate(onlyfiles[:show_n]):
    
        table = pd.read_csv(main.datasets[dataset] + file)
        X = np.array(table.iloc[:, 0])
    
        y = np.array(table.iloc[:, -1])
        y_pred = model(X_plot, parameters[idx])
        
        ax.scatter(X, y, alpha=scatter_alpha)
        ax.plot(X_plot, y_pred, alpha=0.6, linewidth=4)

    ax.set_xlim(xlim)
    ax.set_ylim(miny - margin, maxy + margin)


def plot_best_solutions():

    chi2s = summary_tables('chi2')
    chi2s_test = summary_tables('chi2_test')
    expressions = summary_tables('expression')
    parameters = summary_tables('parameters')

    for dataset in datasets:
        
        best_config, best_method = overall_best(chi2s[dataset])
        
        model = main.make_function(expressions[dataset].loc[best_config, best_method])
        best_parameters = parameters[dataset].loc[best_config, best_method]

        plot_solution(dataset, model, best_parameters, show_n=10)
        
def plot_all_bests(dataset, xlabel='', ylabel='', title='', method_pos=(.98, .03), show_n=7,
                   horizontalalignment='right',scatter_alpha=0.6, xlim=None):

    chi2s = summary_tables('chi2')[dataset]
    chi2s_test = summary_tables('chi2_test')[dataset]
    expressions = summary_tables('expression')[dataset]
    parameters = summary_tables('parameters')[dataset]
    bests = np.argmin(chi2s, axis=0)

    fig, axes = plt.subplots(2,2, figsize=(12, 6), )
    
    for idx, method in enumerate(methods):

        ax = axes.flat[idx]
        
        best_idx = int(bests[idx])
        expression = expressions[method].iloc[best_idx]
        chi2 = chi2s[method].iloc[best_idx]
        chi2_test = chi2s_test[method].iloc[best_idx]

        latex_expression = generate_latex(expression)
        

        model = main.make_function(expression)
        best_parameters = parameters[method].iloc[best_idx]
        
        whowasbest = expressions[method].index[best_idx]
        print(method, ":", whowasbest)
        
        _ = generate_latex(expressions[method][whowasbest], save=f"{dataset}_{method}_{whowasbest}")

        plot_solution(ax, dataset, model, best_parameters, show_n=show_n,
                         title=method, scatter_alpha=scatter_alpha, xlim=xlim)

        if (idx==0) | (idx==1):
            ax.tick_params(axis='x',which='both', bottom=False,top=False,labelbottom=False)
            
        if (idx==1) | (idx==3):
            ax.tick_params(axis='y',which='both', left=False,right=False,labelleft=False)

        if method == 'PhySO':
            method = '$\\phi$-SO'
        ax.text(method_pos[0], method_pos[1], f"{method}\n$MSE_{{train}}$ = {chi2:.4f}\n$MSE_{{test}}$ = {chi2_test:.4f}",
                transform=ax.transAxes, horizontalalignment=horizontalalignment, verticalalignment='bottom', fontsize=16)
        
    fig.tight_layout(pad=1)
    fig.text(0.5, -0.02, xlabel, ha='center', fontsize=18)
    fig.text(-0.02, 0.5, ylabel, va='center', rotation='vertical', fontsize=18)

    plt.suptitle(title, y=1.03, fontsize=20)

    return (fig, axes)



def generate_latex(expr, save=None):

    import string
    alphabet = list(string.ascii_uppercase)
    
    for i in range(10):
        expr = expr.replace(f't[{i}]', alphabet[i])

    # Define symbols
    X, A, B, C, D = sp.symbols('X A B C D')

    # Replace NumPy functions with their SymPy equivalents
    expr = expr.replace("x[:, 0]", 'X')
    expr = expr.replace("np.abs", "Abs")  # Replace np.abs with SymPy's Abs
    expr = expr.replace("np.exp", "exp")  # Replace np.exp with SymPy's exp
    expr = expr.replace("np.log", "log")  # Replace np.log with SymPy's log
    expr = expr.replace("np.sqrt", "sqrt")  # Replace np.sqrt with SymPy's sqrt
    expr = expr.replace("np.square", "square")  # Replace np.sqrt with SymPy's sqrt


    # Convert the string to a SymPy expression without simplification
    sympy_expr = sp.sympify(expr, locals={
        "Abs": sp.Abs,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "square": np.square,
        "X": X, "A": A, "B": B, "C": C, "D": D
    }, evaluate=False)

    # Convert the expression to LaTeX format
    latex_expr = sp.latex(sympy_expr, mul_symbol='dot')

    if save is not None:
        f = open(f"{save}.txt", "w")
        f.write(latex_expr)
        f.close()

    latex_expr = latex_expr.replace("\\left(", "(")
    latex_expr = latex_expr.replace("\left(", "(")
    latex_expr = latex_expr.replace("\\right)", ")")
    latex_expr = latex_expr.replace("\right)", ")")

    return latex_expr


def plot_for_experts(dataset, show_n=12):

    chi2s = summary_tables('chi2')
    chi2s_test = summary_tables('chi2_test')
    expressions = summary_tables('expression')
    parameters = summary_tables('parameters')
    
    plt.rcParams['text.usetex'] = True
    
    
    for idx1, method in enumerate(methods):
        for idx2, config in enumerate(configs):
                
            model = main.make_function(expressions[dataset].loc[config, method])
            best_parameters = parameters[dataset].loc[config, method]
            fig, ax = plt.subplots()
            plot_solution(ax, dataset, model, best_parameters, show_n=show_n)
    
            fig.set_size_inches(16.5, 8.5)
            chi2 = chi2s[dataset].loc[config, method]
            latex_expr = generate_latex(expressions[dataset].loc[config, method])
            ax.set_title(fr"{method} {config} (chi2={chi2:.4f}) : ${latex_expr}$", fontsize=25)
    
            fig.savefig(f"analysis/plots_for_experts/{dataset}_{method}_{config}.png",bbox_inches='tight')