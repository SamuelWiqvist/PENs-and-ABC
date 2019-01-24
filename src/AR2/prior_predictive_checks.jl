# script to generate the parameter data pairs used to run ABC-RS for the
# different methods

#=
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
=#

# load files
include(pwd()*"/src/MA2/set_up.jl")
include(pwd()"/src/generate training test data/generate_parameter_data_pairs.jl")

nbr_obs = 150

srand(1337)
parameters, data = generate_parameter_data_pairs(nbr_obs, sample_from_prior, generate_data)

PyPlot.figure()
PyPlot.plot(y_real)

PyPlot.figure()

for i = 1:nbr_obs
    PyPlot.plot(data[i,:])
end

PyPlot.plot(y_real, "k", linewidth=5)
