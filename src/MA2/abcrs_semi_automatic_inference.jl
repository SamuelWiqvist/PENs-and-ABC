# Run ABC-RS using the sufficient statistics as summary stats

using Pkg
using PyPlot
using Random
using MultivariateStats

include(pwd()*"/src/MA2/set_up.jl")
include(pwd()*"/src/abc algorithms/abc_rs.jl")
include(pwd()*"/src/utilities/posteriorinference.jl")
include(pwd()*"/src/generate training test data/generate_parameter_data_pairs.jl")

# load stored parameter data paris
proposalas = Matrix(CSV.read("data/MA2/abc_data_parameters.csv", allowmissing=:auto))

datasets = Matrix(CSV.read("data/MA2/abc_data_data.csv", allowmissing=:auto))

# Computes the summary statistics"
calc_summary(y_sim::Vector, y_obs::Vector) = autocov(y_sim, 1:2)

# distance function
ρ(s::Vector, s_star::Vector) = euclidean_dist(s, s_star, ones(2))


# pre-run
Random.seed!(1234)
prop_pre_run, data_pre_run = generate_parameter_data_pairs(10^5,sample_from_prior, generate_data)

accapted_proposals,accapted_data = @time abcrs(y_obs, prop_pre_run, data_pre_run, calc_summary, ρ, return_data = true)

# Plot posterior
PyPlot.figure()
PyPlot.plot((0,-2),(-1,1), "g")
PyPlot.plot((-2,2),(1,1), "g")
PyPlot.plot((0,2),(-1,1), "g")
PyPlot.scatter(accapted_proposals[:,1],accapted_proposals[:,2])
PyPlot.plot(θ_true[1],θ_true[2], "k*")
PyPlot.xlabel(L"$\theta_1$")
PyPlot.ylabel(L"$\theta_2$")

# fit linear regresion model
X = [accapted_data accapted_data.^2 accapted_data.^3 accapted_data.^4]

# TODO: fix the lin-reg problems!!


lin_reg_model_1 = llsq(X, accapted_proposals[:,1], trans = false)

lin_reg_model_2 = llsq(X, accapted_proposals[:,2], trans = false)

# Computes the summary statistics"
function calc_summary(y_sim::Vector, y_obs::Vector)

    s1 = [1 y_sim' y_sim'.^2 y_sim'.^3 y_sim'.^4]*lin_reg_model_1
    s2 = [1 y_sim' y_sim'.^2 y_sim'.^3 y_sim'.^4]*lin_reg_model_2

    return [s1,s2]

end

# distance function
ρ(s::Vector, s_star::Vector) = euclidean_dist(s, s_star, ones(2))

# run ABC-RS
appox_post = abcrs(proposalas, datasets, calc_summary, ρ)

# calc posterior quantile interval
posterior_quantile_interval = quantile_interval(appox_post)

# Plot posterior
PyPlot.figure()
PyPlot.plot((0,-2),(-1,1), "g")
PyPlot.plot((-2,2),(1,1), "g")
PyPlot.plot((0,2),(-1,1), "g")
PyPlot.scatter(appox_post[:,1],appox_post[:,2])
PyPlot.plot(θ_true[1],θ_true[2], "k*")
PyPlot.xlabel(L"$\theta_1$")
PyPlot.ylabel(L"$\theta_2$")

# save approx posterior samples
CSV.write("data/MA2/abcrs_post_semi_automatic.csv", DataFrame(appox_post))
