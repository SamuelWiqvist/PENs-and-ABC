# Run ABC-RS using the simple DNN model to calc the summary statistics

# Importantly! We will not use the TensorFlow framework here, instead we run everyhting
# in native Julia

using Distributions
using StatsBase
using KernelDensity
using PyPlot
using Dates

include(pwd()*"/src/g and k dist/generate_parameter_data_for_abc_inference.jl")
include(pwd()*"/src/g and k dist/set_up.jl")
include(pwd()"/src/abc algorithms/abc_rs.jl")
include(pwd()"/src/nets/DNN_forward_pass.jl")
include(pwd()*"/src/utilities/posteriorinference.jl")

# cut off values for data
low_cutoff = -10
high_cutoff = 50

remove_outliers!(datasets, low_cutoff, high_cutoff)
remove_outliers!(y_obs, low_cutoff, high_cutoff)


# load weigths and biases
W_h1 = Matrix(CSV.read("data/gandk/DNN for gandk model iid data/W_h1.csv", allowmissing=:auto))
W_h2 = Matrix(CSV.read("data/gandk/DNN for gandk model iid data/W_h2.csv", allowmissing=:auto))
W_h3 = Matrix(CSV.read("data/gandk/DNN for gandk model iid data/W_h3.csv", allowmissing=:auto))
W_out = Matrix(CSV.read("data/gandk/DNN for gandk model iid data/W_out.csv", allowmissing=:auto))

b1 = Matrix(CSV.read("data/gandk/DNN for gandk model iid data/b1.csv", allowmissing=:auto))
b2 = Matrix(CSV.read("data/gandk/DNN for gandk model iid data/b2.csv", allowmissing=:auto))
b3 = Matrix(CSV.read("data/gandk/DNN for gandk model iid data/b3.csv", allowmissing=:auto))
b_out = Matrix(CSV.read("data/gandk/DNN for gandk model iid data/b_out.csv", allowmissing=:auto))

W_matricies = (W_h1,W_h2,W_h3,W_out)

b_vectors = (b1,b2,b3,b_out)

# function to calc summary stats
function calc_summary(y_sim::Vector, y_obs::Vector)

    remove_outliers!(y_sim, low_cutoff, high_cutoff)
    return forward_pass(y_sim)[:]

end
# distance function
ρ(s::Vector, s_star::Vector) = euclidean_dist(s, s_star, ones(4))

# run ABC-RS
approx_posterior_samples = abcrs(y_obs, proposalas, datasets, calc_summary, ρ)

# calc posterior quantile interval
posterior_quantile_interval = quantile_interval(approx_posterior_samples)

# plot marginal posterior distributions

# calc grid for prior dist
x_grid = -0.5:0.01:10.5

# calc prior dist
priordensity1 = pdf.(Gamma(2,1), x_grid)
priordensity2 = pdf.(Gamma(2,1), x_grid)
priordensity3 = pdf.(Gamma(2,0.5), x_grid)
priordensity4 = pdf.(Gamma(2,1), x_grid)


h1 = kde(approx_posterior_samples[:,1])
h2 = kde(approx_posterior_samples[:,2])
h3 = kde(approx_posterior_samples[:,3])
h4 = kde(approx_posterior_samples[:,4])

PyPlot.figure()
subplot(221)
PyPlot.plot(h1.x,h1.density, "b")
PyPlot.plot(x_grid,priordensity1, "g")
PyPlot.plot((θ_true[1], θ_true[1]), (0, maximum(h1.density)), "k")
PyPlot.ylabel(L"Density")
PyPlot.xlabel(L"$A$")
subplot(222)
PyPlot.plot(h2.x,h2.density, "b")
PyPlot.plot(x_grid,priordensity2, "g")
PyPlot.plot((θ_true[2], θ_true[2]), (0, maximum(h2.density)), "k")
PyPlot.xlabel(L"$B$")
subplot(223)
PyPlot.plot(h3.x,h3.density, "b")
PyPlot.plot(x_grid,priordensity3, "g")
PyPlot.plot((θ_true[3], θ_true[3]), (0, maximum(h3.density)), "k")
PyPlot.xlabel(L"$g$")
PyPlot.ylabel(L"Density")
subplot(224)
PyPlot.plot(h4.x,h4.density, "b")
PyPlot.plot(x_grid,priordensity4, "g")
PyPlot.plot((θ_true[4], θ_true[4]), (0, maximum(h4.density)), "k")
PyPlot.xlabel(L"$k$")


# save approx posterior samples
CSV.write("data/gandk/simple_dnn_abcrs_post_iid_data.csv", DataFrame(approx_posterior_samples))
