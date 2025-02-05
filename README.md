This github is a python wrapper of the Symbolic Regression method based equality graph, eggp, proposed by the paper ["Improving Genetic Programming for Symbolic Regression with Equality Graphs"](https://arxiv.org/abs/2501.17848). This implementation includes a multi-view approach of the data, as explained in the paper ["Multi-View Symbolic Regression"](https://arxiv.org/abs/2402.04298). The goal of MvSR is to discover parametric equations able to describe an ensemble of datasets, rather than generating a single solution to a single dataset. This wrapper is focused around the MvSR aspect of eggp. Very few steps are required to use the method.


Start by cloning the repository. Inside, download the following executable and rename the file to ```eggp```.


https://github.com/folivetti/srtree/releases/download/v2.0.1.0/egraphGP-2.0.1.0-Linux-ghc-9.10.1


Then install the required python packages by running:

```sh
pip install -r requirements.txt
```

You are ready to use eggp MvSR. 
