# script to generate the parameter data pairs used to run ABC-RS for the
# different methods

using DataFrames
using CSV
using Random

# load files
include(pwd()*"/src/AR2/set_up.jl")
include(pwd()*"/src/generate training test data/generate_parameter_data_pairs.jl")

nbr_obs = 500000

Random.seed!(1234)

parameters, data = generate_parameter_data_pairs(nbr_obs, sample_from_prior, generate_data)

CSV.write("data/AR2/abc_data_parameters.csv", DataFrame(parameters))
CSV.write("data/AR2/abc_data_data.csv", DataFrame(data))
