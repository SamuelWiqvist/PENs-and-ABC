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

y_obs_ecdf = ecdf(y_obs)(x)

y_obs = remove_outliers(y_obs, low_cutoff, high_cutoff)

CSV.write("data/gandk/y_obs_ecdf.csv", DataFrame([collect(1:100), y_obs_ecdf]))
CSV.write("data/gandk/y_obs.csv", DataFrame([collect(1:1000), y_obs]))

################################################################################
## generate ABC data
################################################################################

nbr_obs = 100000

Random.seed!(1234)
proposalas, datasets = generate_parameter_data_pairs(nbr_obs, sample_from_informative_prior, generate_data)

datasets_ecdf = zeros(size(datasets,1),100)

for i = 1:size(datasets,1); datasets_ecdf[i,:] = ecdf(datasets[i,:])(x); end

CSV.write("data/gandk/ABC_datasets_ecdf.csv", DataFrame(datasets_ecdf))

for i in 1:size(datasets,1); datasets[i,:] = remove_outliers(datasets[i,:], low_cutoff, high_cutoff); end

CSV.write("data/gandk/ABC_proposals.csv", DataFrame(proposalas))
CSV.write("data/gandk/ABC_datasets.csv", DataFrame(datasets))

################################################################################
## generate training data
################################################################################

nbr_obs_training = 10^6
nbr_obs_val = div(nbr_obs_training,10)
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

CSV.write("data/gandk/y_test.csv", DataFrame(y_test))


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

for i in 1:size(X_training,1); X_training[i,:] = remove_outliers(X_training[i,:], low_cutoff, high_cutoff); end

for i in 1:size(X_val,1); X_val[i,:] = remove_outliers(X_val[i,:], low_cutoff, high_cutoff); end

for i in 1:size(X_test,1); X_test[i,:] = remove_outliers(X_test[i,:], low_cutoff, high_cutoff); end

X_training = reshape(X_training', (1000,100000,10))
X_val = X_val'
X_test = reshape(X_test', (1000,100000,2))

X_training_ecdf = reshape(X_training_ecdf', (100,100000,10))
X_val_ecdf = X_val_ecdf'
X_test_ecdf = reshape(X_test_ecdf', (100,100000,2))

y_training = reshape(y_training', (4,100000,10))
y_val = y_val'
y_test = reshape(y_test', (4,100000,2))

GC.gc()

for i in 1:10
    println(i)
    CSV.write("data/gandk/X_training"*string(i)*".csv", DataFrame(X_training[:,:,i]'))
    CSV.write("data/gandk/y_training"*string(i)*".csv", DataFrame(y_training[:,:,i]'))
end

for i in 1:2
    println(i)
    CSV.write("data/gandk/X_test"*string(i)*".csv", DataFrame(X_test[:,:,i]'))
    CSV.write("data/gandk/y_test"*string(i)*".csv", DataFrame(y_test[:,:,i]'))
end

CSV.write("data/gandk/X_val.csv", DataFrame(X_val'))
CSV.write("data/gandk/y_val.csv", DataFrame(y_val'))

for i in 1:10
    println(i)
    CSV.write("data/gandk/X_training_ecdf"*string(i)*".csv", DataFrame(X_training_ecdf[:,:,i]'))
end

for i in 1:2
    println(i)
    CSV.write("data/gandk/X_test_ecdf"*string(i)*".csv", DataFrame(X_test_ecdf[:,:,i]'))
end

CSV.write("data/gandk/X_val_ecdf.csv", DataFrame(X_val_ecdf'))
