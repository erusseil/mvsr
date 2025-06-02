This github is a frozen version of the analysis performed in Russeil et al. 2025: Exploring Multi -view Symbolic Regression methods in physical sciences. It contains all datasets used for the analysis as well and the scripts to run the analysis and plot the results. In practice, the datasets and MvSR implementation to run on are specified inside `start.sh`. But because, the analysis attempts to compare 4 different codes. In order to everything we installed four separate python environments that we activate when the corresponding analysis is run.

1. PySR version 1.3.1
2. PhySO version 1.1.0a0
3. pyoperon MvSR is installed following the instructions: https://github.com/erusseil/MvSR-analysis
4. eggp should be installed from the `old_eggp` folder. *WARNING* the project was developed before any python wrapper was proposed for eggp. It is now available on PyPi (see official repo: https://github.com/folivetti/eggp) but it is not compatible with this code.
