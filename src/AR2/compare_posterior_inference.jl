
using PyPlot
using DataFrames
using CSV
using KernelDensity
using Distributions
using Statistics

include(pwd()*"/src/MA2/set_up.jl")
include(pwd()*"/src/utilities/posteriorinference.jl")

posterior_exact = Matrix(CSV.read("data/MA2/exact_mcmc_post.csv"; allowmissing=:auto))'
posterior_summary_stats = Matrix(CSV.read("data/MA2/abcrs_post.csv"; allowmissing=:auto))'
posterior_simple_DNN = Matrix(CSV.read("data/MA2/DNN_simple_abcrs_post.csv"; allowmissing=:auto))'
posterior_cnn_DNN = Matrix(CSV.read("data/MA2/cnn_abcrs_post.csv"; allowmissing=:auto))'
posterior_bp_deepsets = Matrix(CSV.read("data/MA2/bp_deepsets_abcrs_post.csv"; allowmissing=:auto))'

# excat mcmc

print(quantile_interval(posterior_exact))

print(calc_summary_stats_for_posterior(posterior_exact))

z_exact = kde((posterior_exact[1,:], posterior_exact[2,:]))

h1_exact = kde(posterior_exact[1,:])
h2_exact = kde(posterior_exact[2,:])

PyPlot.figure()
PyPlot.plot((0,-2),(-1,1), "g")
PyPlot.plot((-2,2),(1,1), "g")
PyPlot.plot((0,2),(-1,1), "g")
PyPlot.scatter(posterior_exact[1,:],posterior_exact[2,:])
PyPlot.plot(θ_true[1],θ_true[2], "k*")
PyPlot.xlabel(L"$\theta_1$")
PyPlot.ylabel(L"$\theta_2$")

PyPlot.figure()
PyPlot.plot((0,-2),(-1,1), "g")
PyPlot.plot((-2,2),(1,1), "g")
PyPlot.plot((0,2),(-1,1), "g")
PyPlot.plt[:contour](z_exact.x, z_exact.y, z_exact.density)
PyPlot.plot(θ_true[1],θ_true[2], "k*")
PyPlot.xlabel(L"$\theta_1$")
PyPlot.ylabel(L"$\theta_2$")

PyPlot.figure()
PyPlot.subplot(121)
PyPlot.plot(h1_exact.x,h1_exact.density, "b")
PyPlot.plot((θ_true[1], θ_true[1]), (0, maximum(h1_exact.density)), "k")
PyPlot.xlabel(L"$\theta_1$")
PyPlot.subplot(122)
PyPlot.plot(h2_exact.x,h2_exact.density, "b")
PyPlot.plot((θ_true[2], θ_true[2]), (0, maximum(h2_exact.density)), "k")
PyPlot.xlabel(L"$\theta_2$")


# summary statistics

print(quantile_interval(posterior_summary_stats))

print(calc_summary_stats_for_posterior(posterior_summary_stats))

z_summary = kde((posterior_summary_stats[1,:], posterior_summary_stats[2,:]))

h1_summary = kde(posterior_summary_stats[1,:])
h2_summary = kde(posterior_summary_stats[2,:])


PyPlot.figure()
PyPlot.plot((0,-2),(-1,1), "g")
PyPlot.plot((-2,2),(1,1), "g")
PyPlot.plot((0,2),(-1,1), "g")
PyPlot.scatter(posterior_summary_stats[1,:],posterior_summary_stats[2,:])
PyPlot.plt[:contour](z_exact.x, z_exact.y, z_exact.density)
PyPlot.plot(θ_true[1],θ_true[2], "k*")
PyPlot.xlabel(L"$\theta_1$")
PyPlot.ylabel(L"$\theta_2$")


PyPlot.figure()
PyPlot.subplot(121)
PyPlot.plot(h1_summary.x,h1_summary.density, "b")
PyPlot.plot(h1_exact.x,h1_exact.density, "b--")
PyPlot.plot((θ_true[1], θ_true[1]), (0, maximum(h1_exact.density)), "k")
PyPlot.xlabel(L"$\theta_1$")
PyPlot.subplot(122)
PyPlot.plot(h2_summary.x,h2_summary.density, "b")
PyPlot.plot(h2_exact.x,h2_exact.density, "b--")
PyPlot.plot((θ_true[2], θ_true[2]), (0, maximum(h2_exact.density)), "k")
PyPlot.xlabel(L"$\theta_2$")


# DNN

print(quantile_interval(posterior_simple_DNN))

print(calc_summary_stats_for_posterior(posterior_simple_DNN))

z_dnn = kde((posterior_simple_DNN[1,:], posterior_simple_DNN[2,:]))

h1_DNN = kde(posterior_simple_DNN[1,:])
h2_DNN = kde(posterior_simple_DNN[2,:])

PyPlot.figure()
PyPlot.plot((0,-2),(-1,1), "g")
PyPlot.plot((-2,2),(1,1), "g")
PyPlot.plot((0,2),(-1,1), "g")
PyPlot.scatter(posterior_simple_DNN[1,:],posterior_simple_DNN[2,:])
PyPlot.plt[:contour](z_exact.x, z_exact.y, z_exact.density)
PyPlot.plot(θ_true[1],θ_true[2], "k*")
PyPlot.xlabel(L"$\theta_1$")
PyPlot.ylabel(L"$\theta_2$")

PyPlot.figure()
PyPlot.subplot(121)
PyPlot.plot(h1_DNN.x,h1_DNN.density, "b")
PyPlot.plot(h1_exact.x,h1_exact.density, "b--")
PyPlot.plot((θ_true[1], θ_true[1]), (0, maximum(h1_exact.density)), "k")
PyPlot.xlabel(L"$\theta_1$")
PyPlot.subplot(122)
PyPlot.plot(h2_DNN.x,h2_DNN.density, "b")
PyPlot.plot(h2_exact.x,h2_exact.density, "b--")
PyPlot.plot((θ_true[2], θ_true[2]), (0, maximum(h2_exact.density)), "k")
PyPlot.xlabel(L"$\theta_2$")

# CNN

print(quantile_interval(posterior_cnn_DNN))

print(calc_summary_stats_for_posterior(posterior_cnn_DNN))

z_cnn_dnn = kde((posterior_cnn_DNN[1,:], posterior_cnn_DNN[2,:]))

h1_cnn = kde(posterior_cnn_DNN[1,:])
h2_cnn = kde(posterior_cnn_DNN[2,:])

PyPlot.figure()
PyPlot.plot((0,-2),(-1,1), "g")
PyPlot.plot((-2,2),(1,1), "g")
PyPlot.plot((0,2),(-1,1), "g")
PyPlot.scatter(posterior_cnn_DNN[1,:],posterior_cnn_DNN[2,:])
PyPlot.plt[:contour](z_exact.x, z_exact.y, z_exact.density)
PyPlot.plot(θ_true[1],θ_true[2], "k*")
PyPlot.xlabel(L"$\theta_1$")
PyPlot.ylabel(L"$\theta_2$")


PyPlot.figure()
PyPlot.subplot(121)
PyPlot.plot(h1_cnn.x,h1_cnn.density, "b")
PyPlot.plot(h1_exact.x,h1_exact.density, "b--")
PyPlot.plot((θ_true[1], θ_true[1]), (0, maximum(h1_exact.density)), "k")
PyPlot.xlabel(L"$\theta_1$")
PyPlot.subplot(122)
PyPlot.plot(h2_cnn.x,h2_cnn.density, "b")
PyPlot.plot(h2_exact.x,h2_exact.density, "b--")
PyPlot.plot((θ_true[2], θ_true[2]), (0, maximum(h2_exact.density)), "k")
PyPlot.xlabel(L"$\theta_2$")

# BP-Deepsets

print(quantile_interval(posterior_bp_deepsets))

print(calc_summary_stats_for_posterior(posterior_bp_deepsets))

z_bp_deepsets = kde((posterior_bp_deepsets[1,:], posterior_bp_deepsets[2,:]))

h1_bp_deepsets = kde(posterior_bp_deepsets[1,:])
h2_bp_deepsets = kde(posterior_bp_deepsets[2,:])

PyPlot.figure()
PyPlot.plot((0,-2),(-1,1), "g")
PyPlot.plot((-2,2),(1,1), "g")
PyPlot.plot((0,2),(-1,1), "g")
PyPlot.scatter(posterior_bp_deepsets[1,:],posterior_bp_deepsets[2,:])
PyPlot.plt[:contour](z_exact.x, z_exact.y, z_exact.density)
PyPlot.plot(θ_true[1],θ_true[2], "k*")
PyPlot.xlabel(L"$\theta_1$")
PyPlot.ylabel(L"$\theta_2$")

PyPlot.figure()
PyPlot.subplot(121)
PyPlot.plot(h1_bp_deepsets.x,h1_bp_deepsets.density, "b")
PyPlot.plot(h1_exact.x,h1_exact.density, "b--")
PyPlot.plot((θ_true[1], θ_true[1]), (0, maximum(h1_exact.density)), "k")
PyPlot.xlabel(L"$\theta_1$")
PyPlot.subplot(122)
PyPlot.plot(h2_bp_deepsets.x,h2_bp_deepsets.density, "b")
PyPlot.plot(h2_exact.x,h2_exact.density, "b--")
PyPlot.plot((θ_true[2], θ_true[2]), (0, maximum(h2_exact.density)), "k")
PyPlot.xlabel(L"$\theta_2$")
