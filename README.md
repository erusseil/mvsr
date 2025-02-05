This github is a python wrapper of the Symbolic Regression method based equality graph, eggp, proposed by the paper ["Improving Genetic Programming for Symbolic Regression with Equality Graphs"](https://arxiv.org/abs/2501.17848). This implementation includes a multi-view approach of the data, as explained in the paper ["Multi-View Symbolic Regression"](https://arxiv.org/abs/2402.04298). The goal of MvSR is to discover parametric equations able to describe an ensemble of datasets, rather than generating a single solution to a single dataset. This wrapper is focused around the MvSR aspect of eggp. Very few steps are required to use the method.


Start by cloning the repository. Inside, download the following executable and rename the file to ```eggp```


https://github.com/folivetti/srtree/releases/download/v2.0.1.0/egraphGP-2.0.1.0-Linux-ghc-9.10.1


Then run:
```sh
chmod +x eggp
```


Finally install the required python packages by running:

```sh
pip install -r requirements.txt
```

You are ready to use eggp MvSR. You can test it by running this simple example, which recovers a polynomial function from a set of three views:


```python
import multiview as mv

MvSR = mv.MvSR("datasets/", max_length=10, pop_size=100, generations=100, n_params=3)
MvSR.run()
MvSR.plot_all_fits()
result_table = MvSR.raw_results
```

As shown in the example, you simply need to provide the path to a folder containing all the views for a given problem. They should be csv files with a specific formating. The columns should have a header indicating their name. The first columns correspond to your explanatory variables (X0, X1, ...) while the last column corresponds to the response variable (Y). In the code snippet, ```MvSR.plot_all_fits()``` offers a fast way to have an overview of the model's ability to fit the views. However you can find all the outputs of the symbolic regression run inside `MvSR.raw_results`.

