# Run ABC-RS using the sufficient statistics as summary stats
using Pkg
#using PyPlot

include(pwd()*"/src/MA2/set_up.jl")
include(pwd()*"/src/abc algorithms/abc_rs.jl")
include(pwd()*"/src/utilities/posteriorinference.jl")

# load stored parameter data paris
proposalas = Matrix(CSV.read("data/MA2/abc_data_parameters.csv", allowmissing=:auto))

datasets = Matrix(CSV.read("data/MA2/abc_data_data.csv", allowmissing=:auto))

# Computes the summary statistics"
calc_summary(y_sim::Vector, y_obs::Vector) = autocov(y_sim, 1:2)

# distance function
ρ(s::Vector, s_star::Vector) = euclidean_dist(s, s_star, ones(2))

# run ABC-RS
approx_posterior_samples = @time abcrs(y_obs, proposalas, datasets, calc_summary, ρ; cutoff_percentile=0.02)

if write_to_files == 1
    # save approx posterior samples
    CSV.write("data/MA2/abcrs_post.csv", DataFrame(approx_posterior_samples))
end
