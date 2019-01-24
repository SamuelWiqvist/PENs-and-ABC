
using PyPlot
using DataFrames
using CSV
using KernelDensity
using Distributions
using Statistics

include(pwd()*"/src/utilities/multivar_cramer.jl")
include(pwd()*"/src/MA2/set_up.jl")
include(pwd()*"/src/utilities/posteriorinference.jl")

# load exact posterior
posterior_exact = Array(Matrix(CSV.read("data/MA2/exact_mcmc_post.csv"; allowmissing=:auto))')

# Simple DNN
posterior_datasize_1 = Matrix(CSV.read("data/MA2/posteriors_multiple_data_sets_abcrs.csv"; allowmissing=:auto))
posterior_datasize_2 = Matrix(CSV.read("data/MA2/posteriors_multiple_data_sets_abcrs.csv"; allowmissing=:auto))
posterior_datasize_3 = Matrix(CSV.read("data/MA2/posteriors_multiple_data_sets_abcrs.csv"; allowmissing=:auto))
posterior_datasize_4 = Matrix(CSV.read("data/MA2/posteriors_multiple_data_sets_abcrs.csv"; allowmissing=:auto))

test_stats_simple_dnn = zeros(4)

test_stats_simple_dnn[1] = multvar_cramer_stat(posterior_datasize_1,posterior_exact[:,1:500])
test_stats_simple_dnn[2] = multvar_cramer_stat(posterior_datasize_1,posterior_exact[:,1:500])
test_stats_simple_dnn[3] = multvar_cramer_stat(posterior_datasize_1,posterior_exact[:,1:500])
test_stats_simple_dnn[4] = multvar_cramer_stat(posterior_datasize_1,posterior_exact[:,1:500])

# Simple CNN
posterior_datasize_1 = Matrix(CSV.read("data/MA2/posteriors_multiple_data_sets_abcrs.csv"; allowmissing=:auto))
posterior_datasize_2 = Matrix(CSV.read("data/MA2/posteriors_multiple_data_sets_abcrs.csv"; allowmissing=:auto))
posterior_datasize_3 = Matrix(CSV.read("data/MA2/posteriors_multiple_data_sets_abcrs.csv"; allowmissing=:auto))
posterior_datasize_4 = Matrix(CSV.read("data/MA2/posteriors_multiple_data_sets_abcrs.csv"; allowmissing=:auto))

test_stats_cnn = zeros(4)

test_stats_cnn[1] = multvar_cramer_stat(posterior_datasize_1,posterior_exact[:,1:500])
test_stats_cnn[2] = multvar_cramer_stat(posterior_datasize_1,posterior_exact[:,1:500])
test_stats_cnn[3] = multvar_cramer_stat(posterior_datasize_1,posterior_exact[:,1:500])
test_stats_cnn[4] = multvar_cramer_stat(posterior_datasize_1,posterior_exact[:,1:500])

# Simple BP-Deepsets
posterior_datasize_1 = Matrix(CSV.read("data/MA2/posteriors_multiple_data_sets_abcrs.csv"; allowmissing=:auto))
posterior_datasize_2 = Matrix(CSV.read("data/MA2/posteriors_multiple_data_sets_abcrs.csv"; allowmissing=:auto))
posterior_datasize_3 = Matrix(CSV.read("data/MA2/posteriors_multiple_data_sets_abcrs.csv"; allowmissing=:auto))
posterior_datasize_4 = Matrix(CSV.read("data/MA2/posteriors_multiple_data_sets_abcrs.csv"; allowmissing=:auto))

test_stats_bp_deepsets = zeros(4)

test_stats_bp_deepsets[1] = multvar_cramer_stat(posterior_datasize_1,posterior_exact[:,1:500])
test_stats_bp_deepsets[2] = multvar_cramer_stat(posterior_datasize_1,posterior_exact[:,1:500])
test_stats_bp_deepsets[3] = multvar_cramer_stat(posterior_datasize_1,posterior_exact[:,1:500])
test_stats_bp_deepsets[4] = multvar_cramer_stat(posterior_datasize_1,posterior_exact[:,1:500])

# Plotting
