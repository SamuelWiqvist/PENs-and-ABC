
using PyPlot
using DataFrames
using CSV
using KernelDensity
using Distributions
using Statistics

include(pwd()*"/src/utilities/multivar_cramer.jl")
include(pwd()*"/src/MA2/set_up.jl")
include(pwd()*"/src/utilities/posteriorinference.jl")

θ_1_true, θ_2_true = θ_true

# load exact posterior
posterior_exact = Array(Matrix(CSV.read("data/MA2/exact_mcmc_post.csv"; allowmissing=:auto))')

θ_1_true_posterior, θ_2_true_posterior = mean(posterior_exact,dims=2)

# ABC

posterior = Matrix(CSV.read("data/MA2/posteriors_multiple_data_sets_abcrs.csv"; allowmissing=:auto))

θ_1_hat = mean(posterior[1:2:end,:],dims=1)
θ_2_hat = mean(posterior[2:2:end,:],dims=1)

# posterior mean loss

sqrt(mean((θ_1_hat.-θ_1_true_posterior).^2))
sqrt(mean((θ_2_hat.-θ_2_true_posterior).^2))

# loss

sqrt(mean((θ_1_hat.-θ_1_true).^2))
sqrt(mean((θ_2_hat.-θ_2_true).^2))

# multivar cramer test

test_stats_abc = zeros(size(posterior,2))
idx = 0
for i in 1:2:size(posterior,1)
    global idx = idx + 1
    global test_stats_abc[idx] = multvar_cramer_stat(posterior[i:i+1,:],posterior_exact[:,1:500])
end

#PyPlot.figure()
#PyPlot.boxplot(test_stats_abc)

# Simple DNN
posterior = Matrix(CSV.read("data/MA2/posteriors_multiple_data_sets_DNN_simple_1.csv"; allowmissing=:auto))

θ_1_hat = mean(posterior[1:2:end,:],dims=1)
θ_2_hat = mean(posterior[2:2:end,:],dims=1)

# posterior mean loss

sqrt(mean((θ_1_hat.-θ_1_true_posterior).^2))
sqrt(mean((θ_2_hat.-θ_2_true_posterior).^2))

# loss

sqrt(mean((θ_1_hat.-θ_1_true).^2))
sqrt(mean((θ_2_hat.-θ_2_true).^2))

# multivar cramer test

test_stats_dnn = zeros(size(posterior,2))
idx = 0
for i in 1:2:size(posterior,1)
    global idx = idx + 1
    global test_stats_dnn[idx] = multvar_cramer_stat(posterior[i:i+1,:],posterior_exact[:,1:500])
end

#PyPlot.figure()
#PyPlot.boxplot(test_stats_abc)


# Simple CNN
posterior = Matrix(CSV.read("data/MA2/posteriors_multiple_data_sets_abcrs.csv"; allowmissing=:auto))

θ_1_hat = mean(posterior[1:2:end,:],dims=1)
θ_2_hat = mean(posterior[2:2:end,:],dims=1)

# posterior mean loss

sqrt(mean((θ_1_hat.-θ_1_true_posterior).^2))
sqrt(mean((θ_2_hat.-θ_2_true_posterior).^2))

# loss

sqrt(mean((θ_1_hat.-θ_1_true).^2))
sqrt(mean((θ_2_hat.-θ_2_true).^2))

# multivar cramer test

test_stats_cnn = zeros(size(posterior,2))
idx = 0
for i in 1:2:size(posterior,1)
    global idx = idx + 1
    global test_stats_cnn[idx] = multvar_cramer_stat(posterior[i:i+1,:],posterior_exact[:,1:500])
end

#PyPlot.figure()
#PyPlot.boxplot(test_stats_cnn)


# Simple BP-Deepsets
posterior = Matrix(CSV.read("data/MA2/posteriors_multiple_data_sets_abcrs.csv"; allowmissing=:auto))

θ_1_hat = mean(posterior[1:2:end,:],dims=1)
θ_2_hat = mean(posterior[2:2:end,:],dims=1)

# posterior mean loss

sqrt(mean((θ_1_hat.-θ_1_true_posterior).^2))
sqrt(mean((θ_2_hat.-θ_2_true_posterior).^2))

# loss

sqrt(mean((θ_1_hat.-θ_1_true).^2))
sqrt(mean((θ_2_hat.-θ_2_true).^2))

# multivar cramer test

test_stats_bp_deepsets = zeros(size(posterior,2))
idx = 0
for i in 1:2:size(posterior,1)
    global idx = idx + 1
    global test_stats_bp_deepsets[idx] = multvar_cramer_stat(posterior[i:i+1,:],posterior_exact[:,1:500])
end

#PyPlot.figure()
#PyPlot.boxplot(test_stats_abc)


# Plotting compare cramer

data = zeros(length(test_stats_abc),2)

data[:,1] = test_stats_abc
data[:,2] = test_stats_dnn

PyPlot.figure()
PyPlot.boxplot(data)
