# Run ABC-RS

#using Distributions
using StatsBase
using CSV
using DataFrames

include(pwd()*"/src/g and k dist/generate_iid_data.jl")
include(pwd()*"/src/g and k dist/set_up.jl")
include(pwd()*"/src/abc algorithms/abc_rs.jl")
include(pwd()*"/src/utilities/posteriorinference.jl")

# Computes the summary statistics
calc_summary(y_sim::Vector, y_obs::Vector) = [percentile(y_sim, [20;40;60;80]);skewness(y_sim)]

# set weigths
w  =  [0.22; 0.19; 0.53; 2.97; 1.90]  # from " Approximate maximum likelihood estimation using data-cloning ABC"

# create distance function
ρ(s::Vector, s_star::Vector) = euclidean_dist(s::Vector, s_star::Vector, w)

# run ABC-RS
approx_posterior_samples = @time abcrs(y_obs, proposalas, datasets, calc_summary, ρ, return_data=false)

if write_to_files == 1
    # save approx posterior samples
    CSV.write("data/gandk/abcrs_post.csv", DataFrame(approx_posterior_samples))
end
