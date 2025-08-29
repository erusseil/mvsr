This github is a frozen version of the analysis performed in Russeil et al. 2025: Exploring Multi -view Symbolic Regression methods in physical sciences. It contains all datasets used for the analysis as well and the scripts to run the analysis and plot the results. In practice, the datasets and MvSR implementation to run on are specified inside `start.sh`. But because, the analysis attempts to compare 4 different codes, we recommand installing four separate python environments that are activated when the corresponding analysis is run.

1. PySR version 1.3.1
2. PhySO version 1.1.10
4. pyoperon MvSR is installed following the instructions: https://github.com/erusseil/MvSR-analysis
5. eggp version 1.0.6

The raw results from running the full analysis can be found inside the ```result```. All plots and result exploration can be found inside the ```process_result.ipynb``` notebook. The figures generated here can be found ```inside analysis/plots```. 
