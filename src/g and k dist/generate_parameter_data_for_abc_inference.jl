
using Pkg
using Random

# load files
include(pwd()*"/src/g and k dist/set_up.jl")
include(pwd()*"/src/generate training test data/generate_parameter_data_pairs.jl")

nbr_obs = 100000

Random.seed!(1234)

proposalas, datasets = generate_parameter_data_pairs(nbr_obs, sample_from_informative_prior, generate_data)

for i in 1:size(datasets,1); remove_outliers!(datasets[i,:], low_cutoff, high_cutoff); end
