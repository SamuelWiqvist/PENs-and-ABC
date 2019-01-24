# prior predictive checks for the alpha-stable dist model

# load packages
using StatsBase
using PyPlot
using KernelDensity
using Distributions

# load files
include(pwd()*"/src/alpha stable dist/set_up.jl")
include(pwd()*"/src/generate training test data/generate_parameter_data_pairs.jl")

# simulate data using wide uniform priors
nbr_obs = 10000

Random.seed!(1234)

proposalas, datasets = generate_parameter_data_pairs(nbr_obs, sample_from_prior, generate_data)

PyPlot.figure()
h1 = kde(y_obs)
PyPlot.plot(h1.x,h1.density, "k", linewidth=5)

summarystats(y_obs)

for i = 1:size(proposalas,1)
    proposalas[i,:] = map_parameters(proposalas[i,:])
end

function plot_ecdf(datasets, y_obs)

    nbr_obs = size(datasets,1)

    PyPlot.figure()
    x = LinRange(-10, 50, 100)

    for i = 1:nbr_obs
        ecdf_func = ecdf(datasets[i,:])
        PyPlot.plot(x,ecdf_func(x))
    end

    ecdf_func = ecdf(y_obs)
    PyPlot.plot(x,ecdf_func(x), "k", linewidth=5)

end


plot_ecdf(datasets[1:5,:], y_obs)


maximum(y_obs)
minimum(y_obs)

data_lower = -10
data_upper = 50

y_obs = remove_outliers(y_obs, data_lower, data_upper)

for i in 1:size(datasets,1); datasets[i,:] = remove_outliers(datasets[i,:], data_lower, data_upper); end

function plot_data(datasets, y_obs)

    nbr_obs = size(datasets,1)

    PyPlot.figure()

    for i = 1:nbr_obs
        #h1 = kde(datasets[i,:]; boundary=(data_lower,data_upper))
        h1 = kde(datasets[i,:])

        PyPlot.plot(h1.x,h1.density)
    end

    h1 = kde(y_obs)
    PyPlot.plot(h1.x,h1.density, "k", linewidth=5)

end


plot_data(datasets[1:5,:], y_obs)


size(datasets)


robust_scaler(datasets)
robust_scaler(y_obs)


minmax_scaler(datasets)
minmax_scaler(y_obs)


PyPlot.figure()
PyPlot.subplot(221)
h = PyPlot.plt[:hist](proposalas[:,1],100)
PyPlot.subplot(222)
h = PyPlot.plt[:hist](proposalas[:,2],100)
PyPlot.subplot(223)
h = PyPlot.plt[:hist](proposalas[:,3],100)
PyPlot.subplot(224)
h = PyPlot.plt[:hist](proposalas[:,4],100)
