This repository is a working code example of the system being described in the paper __"Mining Potentially Explanatory Patterns via Partial Solutions"__, which you can find [here](https://doi.org/10.1145/3638530.3654318) 

The system defines the following components:
  * PS: a partial solution
    *  (essentially a wrapper over a numpy array of integers, where * is represented by -1)
  * PSMiner: The algorithm which searches for nice PSs using a reference population
    *  (used to obtain a __PS Catalog__, which is just a list of PSs) 
  * PickAndMergeSampler: The algorithm which samples from the PS Catalog to obtain "full" solutions

For your convenience, there's also a lengthy list of Benchmark problems for you to try.

(To run this script, you'll need Python 3.10 or later, and to have the following installed:
* numpy
* scikit-learn
* scipy (for the optional BivariateANOVALinkage.py)


***

In case you're struggling to understand the code, here are some descriptions of the classes:
* **PRef**: The reference population, packaged into a convenient object which includes both the solutions and their fitnesses
* **FS** / FullSolution: A wrapper for numpy arrays of integers
* **SearchSpace**: Represents the combinatorial search space we are searching in
* **EvaluatedPS** / **EvaluatedFS**: data structures used to include the fitness value with PSs/FSs
* **Metric**: An object used to evaluate a PS, eg Simplicity, MeanFitness, Atomicity
  *  There are a lot of them to be used, and note that you may replace Atomicity with Linkage etc..
