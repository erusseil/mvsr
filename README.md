This github is a frozen version of the analysis performed in Russeil et al. 2025: Exploring Multi -view Symbolic Regression methods in physical sciences. It contains all datasets used for the analysis as well and the scripts to run the analysis and plot the results. In practice, the datasets and MvSR implementation to run on are specified inside `start.sh`. But because, the analysis attempts to compare 4 different codes, we recommend installing four separate python environments that are activated when the corresponding analysis is run.

1. PySR version 1.3.1
2. PhySO version 1.1.10
3. pyoperon MvSR is installed following the instructions: https://github.com/erusseil/MvSR-analysis
4. eggp version 1.0.6

You should be able to specify the method to use and the datasets to process inside ```start.sh``` and run it with: 
```sh
sh start.sh
```

operon_increased was manually run by modifiying the operon's hyparameters inside the ```main.py``` file. The raw results from running the full analysis can be found inside the ```result```. All plots and result exploration can be found inside the ```process_result.ipynb``` notebook. The figures generated here can be found ```inside analysis/plots```. 
