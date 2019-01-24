# Code for the paper *Partially Exchangeable Networks and Architectures for Learning Summary Statistics in Approximate Bayesian Computation*

This repository contains all the code for the pre-print paper *Partially Exchangeable Networks and Architectures for Learning Summary Statistics in Approximate Bayesian Computation*.

# File structure

/lunarc

* Run-scripts and output files for LUNARC ([http://www.lunarc.lu.se/](http://www.lunarc.lu.se/))

/notebooks

* Notebooks used for creating the plots in the paper (and several other plots)

/src

* source files

/src/abc algorithms

* code for generic ABC rejection sampling algorithm

/src/alpha stable

* source files for the alpha-stable distribtion

/src/AR2

* source files for the AR2 model

/src/g-and-k distribtion

* source files for the g-and-k distribution

/src/generate training test data  

* generic code to generate training and test data for some model

/src/MA2

* source files for the MA2 model

/src/nets

* generic code for the MLP network and generic loss functions

/src/utilities

* help functions (random number generators, distance functions ets.)


# Software

The code is written in Julia 1.0.0 and the framework Knet is used to train the networks. The code can be run on both CPUs and GPUs.

The networks are trained on a Nvidia K80 card by utilizing the GPU nodes at LUNARC ([http://www.lunarc.lu.se/](http://www.lunarc.lu.se/)).

Packages used:

* `Knet` v1.1.0
* `Distributions` v0.16.4
* `MultivariateStats` v0.6.0 master
* `CSV` v0.3.1
* `DataFrames` v0.13.1
* `StatsFuns` v0.7.0
* `StatsBase` v0.25.0

# Data

The data used in the paper can be generated from the code.
