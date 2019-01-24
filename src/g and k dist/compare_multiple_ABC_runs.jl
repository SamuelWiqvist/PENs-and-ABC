
using PyPlot
using DataFrames
using CSV
using KernelDensity
using Distributions
using Statistics

include(pwd()*"/src/utilities/multivar_cramer.jl")
include(pwd()*"/src/g and k dist/set_up.jl")
include(pwd()*"/src/utilities/posteriorinference.jl")

A_true, B_true, g_true, k_true = θ_true

# load exact posterior
posterior_exact = Array(Matrix(CSV.read("data/gandk/exact_inference_post.csv"; allowmissing=:auto))')

A_true_posterior, B_true_posterior, g_true_posterior, k_true_posterior = mean(posterior_exact,dims=2)

# ABC

posterior = Matrix(CSV.read("data/gandk/posteriors_multiple_data_sets_abcrs.csv"; allowmissing=:auto))

A_hat = mean(posterior[1:4:end,:],dims=1)
B_hat = mean(posterior[2:4:end,:],dims=1)
g_hat = mean(posterior[3:4:end,:],dims=1)
k_hat = mean(posterior[4:4:end,:],dims=1)

# posterior mean loss

sqrt(mean((A_hat.-A_true_posterior).^2))
sqrt(mean((B_hat.-B_true_posterior).^2))
sqrt(mean((g_hat.-g_true_posterior).^2))
sqrt(mean((k_hat.-k_true_posterior).^2))

# loss

sqrt(mean((A_hat.-A_true).^2))
sqrt(mean((B_hat.-B_true).^2))
sqrt(mean((g_hat.-g_true).^2))
sqrt(mean((k_hat.-k_true).^2))

# multivar cramer test

test_stats_abc = zeros(size(posterior,2))
idx = 0
for i in 1:4:size(posterior,1)
    global idx = idx + 1
    global test_stats_abc[idx] = multvar_cramer_stat(posterior[i:i+3,:],posterior_exact[:,1:500])
end

PyPlot.figure()
PyPlot.boxplot(test_stats_abc)

# Simple DNN
posterior = Matrix(CSV.read("data/MA2/posteriors_multiple_data_sets_abcrs.csv"; allowmissing=:auto))

posterior = Matrix(CSV.read("data/gandk/posteriors_multiple_data_sets_abcrs.csv"; allowmissing=:auto))

A_hat = mean(posterior[1:4:end,:],dims=1)
B_hat = mean(posterior[2:4:end,:],dims=1)
g_hat = mean(posterior[3:4:end,:],dims=1)
k_hat = mean(posterior[4:4:end,:],dims=1)

# posterior mean loss

sqrt(mean((A_hat.-A_true_posterior).^2))
sqrt(mean((B_hat.-B_true_posterior).^2))
sqrt(mean((g_hat.-g_true_posterior).^2))
sqrt(mean((k_hat.-k_true_posterior).^2))

# loss

sqrt(mean((A_hat.-A_true).^2))
sqrt(mean((B_hat.-B_true).^2))
sqrt(mean((g_hat.-g_true).^2))
sqrt(mean((k_hat.-k_true).^2))

# multivar cramer test

test_stats_abc = zeros(size(posterior,2))
idx = 0
for i in 1:4:size(posterior,1)
    global idx = idx + 1
    global test_stats_abc[idx] = multvar_cramer_stat(posterior[i:i+3,:],posterior_exact[:,1:500])
end

PyPlot.figure()
PyPlot.boxplot(test_stats_abc)

# BP-Deepsets
posterior = Matrix(CSV.read("data/MA2/posteriors_multiple_data_sets_abcrs.csv"; allowmissing=:auto))

posterior = Matrix(CSV.read("data/gandk/posteriors_multiple_data_sets_abcrs.csv"; allowmissing=:auto))

A_hat = mean(posterior[1:4:end,:],dims=1)
B_hat = mean(posterior[2:4:end,:],dims=1)
g_hat = mean(posterior[3:4:end,:],dims=1)
k_hat = mean(posterior[4:4:end,:],dims=1)

# posterior mean loss

sqrt(mean((A_hat.-A_true_posterior).^2))
sqrt(mean((B_hat.-B_true_posterior).^2))
sqrt(mean((g_hat.-g_true_posterior).^2))
sqrt(mean((k_hat.-k_true_posterior).^2))

# loss

sqrt(mean((A_hat.-A_true).^2))
sqrt(mean((B_hat.-B_true).^2))
sqrt(mean((g_hat.-g_true).^2))
sqrt(mean((k_hat.-k_true).^2))

# multivar cramer test

test_stats_abc = zeros(size(posterior,2))
idx = 0
for i in 1:4:size(posterior,1)
    global idx = idx + 1
    global test_stats_abc[idx] = multvar_cramer_stat(posterior[i:i+3,:],posterior_exact[:,1:500])
end

PyPlot.figure()
PyPlot.boxplot(test_stats_abc)
