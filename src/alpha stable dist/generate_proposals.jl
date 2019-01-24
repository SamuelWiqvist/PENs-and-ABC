# script to generate the parameter data pairs used to run ABC-RS for the
# different methods

using Pkg
using Random
using StatsBase
using Distributions

include(pwd()*"/src/alpha stable dist/set_up.jl")
include(pwd()*"/src/generate training test data/generate_parameter_data_pairs.jl")


################################################################################
## generate training data
################################################################################

nbr_obs_training = div(10^6,2)
nbr_obs_val = 5000
nbr_obs_test = 2*10^5

nbr_obs = nbr_obs_training + nbr_obs_test + nbr_obs_val

Random.seed!(1235)
parameters, data = generate_parameter_data_pairs(nbr_obs_training, sample_from_prior, generate_data)

length(findall(isnan, data))

CSV.write("data/alpha stable/y_training.csv", DataFrame(parameters))
#CSV.write("data/multivar alpha stable/X_training.csv", DataFrame(data))

Random.seed!(1236)
parameters, data = generate_parameter_data_pairs(nbr_obs_test, sample_from_prior, generate_data)

length(findall(isnan, data))

CSV.write("data/alpha stable/y_test.csv", DataFrame(parameters))
#CSV.write("data/multivar alpha stable/X_test.csv", DataFrame(data))

Random.seed!(1237)
parameters, data = generate_parameter_data_pairs(nbr_obs_val, sample_from_prior, generate_data)

length(findall(isnan, data))

CSV.write("data/alpha stable/y_val.csv", DataFrame(parameters))
#CSV.write("data/multivar alpha stable/X_val.csv", DataFrame(data))

#CSV.write("data/gandk/y_obs.csv", DataFrame(y_obs))
