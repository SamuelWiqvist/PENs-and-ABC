# prior predictive checks for the g and k dist model

# load packages
using StatsBase
using PyPlot
using KernelDensity

# load files
include(pwd()*"/src/g and k dist/set_up.jl")
include(pwd()*"/src/generate training test data/generate_parameter_data_pairs.jl")


# plot real data
function plot_data(y_obs)

    PyPlot.figure()
    PyPlot.plt[:hist](y_obs,100)

    ecdf_func = ecdf(y_obs)
    x = LinRange(0, 30, 1000)
    # plot emperical CDF
    PyPlot.figure()
    PyPlot.plot(x,ecdf_func(x))

end

# plot data
function plot_sim_datasets_and_data(datasets, y_obs)

    nbr_obs = size(datasets,1)

    PyPlot.figure()

    for i = 1:nbr_obs
        h1 = kde(datasets[i,:])
        PyPlot.plot(h1.x,h1.density)
    end

    h1 = kde(y_obs)
    PyPlot.plot(h1.x,h1.density, "k", linewidth=5)


    PyPlot.figure()
    x = LinRange(-10, 50, 100)

    for i = 1:nbr_obs
        ecdf_func = ecdf(datasets[i,:])
        PyPlot.plot(x,ecdf_func(x))
    end

    ecdf_func = ecdf(y_obs)
    PyPlot.plot(x,ecdf_func(x), "k", linewidth=5)

end


Random.seed!(1234)
y_obs = generate_data(θ_true)

# plot data
plot_data(y_obs)

# simulate data using wide uniform priors
nbr_obs = 10000

Random.seed!(1234)

proposalas, datasets = generate_parameter_data_pairs(nbr_obs, sample_from_prior, generate_data)

plot_sim_datasets_and_data(datasets, y_obs)

# simulate data using (more) informative priors

function sample_from_informative_prior()

    prior_A = rand_gamma(2,1)
    prior_B = rand_gamma(2,1)
    prior_g = rand_gamma(2,0.5)
    prior_k = rand_gamma(2,1)

    return [prior_A;prior_B;prior_g;prior_k]

end

Random.seed!(1234)

proposalas, datasets = generate_parameter_data_pairs(nbr_obs, sample_from_informative_prior, generate_data)

plot_sim_datasets_and_data(datasets[1:5,:], y_obs)

# remove "outliers" form data

low_cutoff = -10
high_cutoff = 50

for i in 1:size(datasets,1); datasets[i,:] = remove_outliers(datasets[i,:], low_cutoff, high_cutoff); end

plot_sim_datasets_and_data(datasets[1:5,:], y_obs)


PyPlot.figure()
PyPlot.subplot(221)
h = PyPlot.plt[:hist](proposalas[:,1],100)
PyPlot.subplot(222)
h = PyPlot.plt[:hist](proposalas[:,2],100)
PyPlot.subplot(223)
h = PyPlot.plt[:hist](proposalas[:,3],100)
PyPlot.subplot(224)
h = PyPlot.plt[:hist](proposalas[:,4],100)
