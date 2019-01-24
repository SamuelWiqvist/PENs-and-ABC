# script to generate the parameter data pairs

using CSV
using DataFrames
using Distributions

# load files
include(pwd()*"/src/alpha stable dist/set_up.jl")
include(pwd()*"/src/generate training test data/generate_parameter_data_pairs.jl")

nbr_obs = 500000

Random.seed!(124)
parameters, data = generate_parameter_data_pairs(nbr_obs, sample_from_prior, generate_data)

# check for nans
length(findall(isnan, data))

CSV.write("data/alpha stable/abc_data_parameters.csv", DataFrame(parameters))
CSV.write("data/alpha stable/abc_data_data.csv", DataFrame(data))

# reshape(data[1,:],2,200) transforms the data back to correct dims
