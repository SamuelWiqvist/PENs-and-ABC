# Code for the paper *Partially Exchangeable Networks and Architectures for Learning Summary Statistics in Approximate Bayesian Computation*

This repository contains the code for the paper *Partially Exchangeable Networks and Architectures for Learning Summary Statistics in Approximate Bayesian Computation* by S. Wiqvist, P-A. Mattei, U. Picchini and J. Frellsen. Link to pre-print version: https://arxiv.org/abs/1901.10230.

**N.B.:** The results in the pre-print at arXiv v2 are computed using the version of the code at tag *preprint_v2*.  

## File structure

/lunarc

* run-scripts and output files for LUNARC ([http://www.lunarc.lu.se/](http://www.lunarc.lu.se/))

/notebooks

* notebooks used for creating the plots in the paper (and several other plots)

/src

* source files

/src/abc algorithms

* code for generic ABC rejection sampling algorithm

/src/alpha stable

* source files for the alpha-stable distribution

/src/AR2

* source files for the AR2 model

/src/g-and-k distribution

* source files for the g-and-k distribution

/src/generate training test data  

* generic code to generate training and test data for some model

/src/MA2 noisy data

* source files for the MA2 model

/src/nets

* generic code for the MLP network and generic loss functions

/src/utilities

* help functions (random number generators, distance functions etc.)


## Software

The code is written in Julia 1.0.0 and the framework Knet is used to train the networks. The code can be run on both CPUs and GPUs.

Some of the posterior inference analyses are carried out in Python 3.6.5 using the package POT: Python Optimal Transport.

Julia packages used:

* `Knet` v1.1.0
* `Distributions` v0.16.4
* `MultivariateStats` v0.6.0 master
* `CSV` v0.3.1
* `DataFrames` v0.13.1
* `StatsFuns` v0.7.0
* `StatsBase` v0.25.0

Python packages used:

* `NumPy` v1.14.3
* `ot` v0.5.1

## Data

The data used in the paper can be generated from the code.
