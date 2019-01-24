# script to generate the parameter data pairs used to run ABC-RS for the
# different methods

using Pkg
using Random
using StatsBase

include(pwd()*"/src/g and k dist/set_up.jl")
include(pwd()*"/src/generate training test data/generate_parameter_data_pairs.jl")

low_cutoff = -10
high_cutoff = 50

cdf_lower = 0
cdf_upper = 50

x = LinRange(cdf_lower, cdf_upper, 100)

################################################################################
## generate data
################################################################################

θ_true = [3; 1; 2; .5]

Random.seed!(1234)
y_obs = generate_data(θ_true)

y_obs = ecdf(y_obs)(x)


################################################################################
## generate ABC data
################################################################################

nbr_obs = 100000

Random.seed!(1234)
proposalas, datasets = generate_parameter_data_pairs(nbr_obs, sample_from_informative_prior, generate_data)

datasets_ecdf = zeros(size(datasets,1),100)

for i = 1:size(datasets,1); datasets_ecdf[i,:] = ecdf(datasets[i,:])(x); end


datasets = datasets_ecdf


################################################################################
## generate training data
################################################################################

nbr_obs_training = div(10^6,2)
nbr_obs_val = 5000
nbr_obs_test = 2*10^5

nbr_obs = nbr_obs_training + nbr_obs_test + nbr_obs_val

Random.seed!(1235)
parameters, data = generate_parameter_data_pairs(nbr_obs_training, sample_from_informative_prior, generate_data)

X_training = data
y_training = parameters

Random.seed!(1236)
parameters, data = generate_parameter_data_pairs(nbr_obs_test, sample_from_informative_prior, generate_data)

X_test = data
y_test = parameters

Random.seed!(1237)
parameters, data = generate_parameter_data_pairs(nbr_obs_val, sample_from_informative_prior, generate_data)

X_val = data
y_val = parameters

X_training_ecdf = zeros(nbr_obs_training,100)
for i = 1:nbr_obs_training; X_training_ecdf[i,:] = ecdf(X_training[i,:])(x); end

X_val_ecdf = zeros(nbr_obs_val,100)
for i = 1:nbr_obs_val; X_val_ecdf[i,:] = ecdf(X_val[i,:])(x); end

X_test_ecdf = zeros(nbr_obs_test,100)
for i = 1:nbr_obs_test; X_test_ecdf[i,:] = ecdf(X_test[i,:])(x); end

X_training = convert(Array{Float32,2}, X_training_ecdf')
X_val = convert(Array{Float32,2}, X_val_ecdf')
X_test = convert(Array{Float32,2}, X_test_ecdf')

y_training = convert(Array{Float32,2}, y_training')
y_val = convert(Array{Float32,2}, y_val')
y_test = convert(Array{Float32,2}, y_test')

GC.gc()

y_obs = convert(Array{Float32,1},y_obs)

proposalas = convert(Array{Float32,2},proposalas)
datasets = convert(Array{Float32,2},datasets)

GC.gc()
GC.gc()
