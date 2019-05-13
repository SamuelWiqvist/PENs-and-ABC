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

#=
using PyPlot
PyPlot.figure()
h = PyPlot.plt[:hist](y_obs,50)
PyPlot.ylabel("Freq.")

pl.ylabel("EMD")
y_obs = remove_outliers(y_obs, low_cutoff, high_cutoff)
=#

GC.gc()


################################################################################
## generate ABC data
################################################################################

nbr_obs = 100000

Random.seed!(1234)
proposalas, datasets = generate_parameter_data_pairs(nbr_obs, sample_from_informative_prior, generate_data)

for i in 1:size(datasets,1); datasets[i,:] = remove_outliers(datasets[i,:], low_cutoff, high_cutoff); end

GC.gc()


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

for i in 1:size(X_training,1); X_training[i,:] = remove_outliers(X_training[i,:], low_cutoff, high_cutoff); end

for i in 1:size(X_val,1); X_val[i,:] = remove_outliers(X_val[i,:], low_cutoff, high_cutoff); end

for i in 1:size(X_test,1); X_test[i,:] = remove_outliers(X_test[i,:], low_cutoff, high_cutoff); end

GC.gc()


X_training = convert(Array{Float32,2}, X_training')
X_val = convert(Array{Float32,2}, X_val')
X_test = convert(Array{Float32,2}, X_test')

y_training = convert(Array{Float32,2}, y_training')
y_val = convert(Array{Float32,2}, y_val')
y_test = convert(Array{Float32,2}, y_test')

GC.gc()

y_obs = convert(Array{Float32,1},y_obs)

proposalas = convert(Array{Float32,2},proposalas)
datasets = convert(Array{Float32,2},datasets)

GC.gc()

CSV.write("data/gandk/y_obs.csv", DataFrame([collect(1:1000), y_obs]))
CSV.write("data/gandk/y_test.csv", DataFrame(y_test'))
