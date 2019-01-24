# script to generate the parameter data pairs used to run ABC-RS for the
# different methods

using Distributions
using StatsBase
using KernelDensity
using PyPlot
using TensorFlow
using Dates
using DataFrames
using MultivariateStats
using JLD
using HDF5

# load files
include(pwd()*"/src/multivar alpha stable/set_up.jl")
include(pwd()*"/src/generate training test data/generate_parameter_data_pairs.jl")

nbr_obs = 500000

parameters, data = generate_parameter_data_pairs(nbr_obs, sample_from_prior, generate_data)

CSV.write("data/MA2/abc_data_parameters.csv", DataFrame(parameters))
CSV.write("data/MA2/abc_data_data.csv", DataFrame(data))
