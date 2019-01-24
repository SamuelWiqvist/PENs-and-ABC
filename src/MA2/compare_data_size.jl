
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
posterior_datasize_1 = Array(Matrix(CSV.read("data/MA2/DNN_simple_1_abcrs_post.csv"; allowmissing=:auto))')
posterior_datasize_2 = Array(Matrix(CSV.read("data/MA2/DNN_simple_2_abcrs_post.csv"; allowmissing=:auto))')
posterior_datasize_3 = Array(Matrix(CSV.read("data/MA2/DNN_simple_3_abcrs_post.csv"; allowmissing=:auto))')
posterior_datasize_4 = Array(Matrix(CSV.read("data/MA2/DNN_simple_4_abcrs_post.csv"; allowmissing=:auto))')

test_stats_simple_dnn = zeros(4)

test_stats_simple_dnn[1] = multvar_cramer_stat(posterior_datasize_1,posterior_exact[:,1:500])
test_stats_simple_dnn[2] = multvar_cramer_stat(posterior_datasize_2,posterior_exact[:,1:500])
test_stats_simple_dnn[3] = multvar_cramer_stat(posterior_datasize_3,posterior_exact[:,1:500])
test_stats_simple_dnn[4] = multvar_cramer_stat(posterior_datasize_4,posterior_exact[:,1:500])

# Simple CNN
posterior_datasize_1 = Array(Matrix(CSV.read("data/MA2/cnn_1_abcrs_post.csv"; allowmissing=:auto))')
posterior_datasize_2 = Array(Matrix(CSV.read("data/MA2/cnn_2_abcrs_post.csv"; allowmissing=:auto))')
posterior_datasize_3 = Array(Matrix(CSV.read("data/MA2/cnn_3_abcrs_post.csv"; allowmissing=:auto))')
posterior_datasize_4 = Array(Matrix(CSV.read("data/MA2/cnn_4_abcrs_post.csv"; allowmissing=:auto))')

test_stats_cnn = zeros(4)

test_stats_cnn[1] = multvar_cramer_stat(posterior_datasize_1,posterior_exact[:,1:500])
test_stats_cnn[2] = multvar_cramer_stat(posterior_datasize_2,posterior_exact[:,1:500])
test_stats_cnn[3] = multvar_cramer_stat(posterior_datasize_3,posterior_exact[:,1:500])
test_stats_cnn[4] = multvar_cramer_stat(posterior_datasize_4,posterior_exact[:,1:500])

# Simple BP-Deepsets
posterior_datasize_1 = Array(Matrix(CSV.read("data/MA2/bp_deepsets_1_abcrs_post.csv"; allowmissing=:auto))')
posterior_datasize_2 = Array(Matrix(CSV.read("data/MA2/bp_deepsets_2_abcrs_post.csv"; allowmissing=:auto))')
posterior_datasize_3 = Array(Matrix(CSV.read("data/MA2/bp_deepsets_3_abcrs_post.csv"; allowmissing=:auto))')
posterior_datasize_4 = Array(Matrix(CSV.read("data/MA2/bp_deepsets_4_abcrs_post.csv"; allowmissing=:auto))')

test_stats_bp_deepsets = zeros(4)

test_stats_bp_deepsets[1] = multvar_cramer_stat(posterior_datasize_1,posterior_exact[:,1:500])
test_stats_bp_deepsets[2] = multvar_cramer_stat(posterior_datasize_2,posterior_exact[:,1:500])
test_stats_bp_deepsets[3] = multvar_cramer_stat(posterior_datasize_3,posterior_exact[:,1:500])
test_stats_bp_deepsets[4] = multvar_cramer_stat(posterior_datasize_4,posterior_exact[:,1:500])

# plotting
x_scale = [10^6, 10^5, 10^4, 10^3]

PyPlot.figure()
PyPlot.semilogx(x_scale, test_stats_simple_dnn, "*-b")
PyPlot.semilogx(x_scale, test_stats_cnn, "*-r")
PyPlot.semilogx(x_scale, test_stats_bp_deepsets, "*-g")
PyPlot.xlabel("Number of obs. in training data")
PyPlot.ylabel("Multi. var Cramer statistics")
