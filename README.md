# Deep learning architectures for learning summary statistics for ABC

Combining ABC and deep learning methods. Using various deep learning architectures to learn summary statistics for ABC.

## Software

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
