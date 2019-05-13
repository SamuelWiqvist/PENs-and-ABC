using CSV
using DataFrames
using Statistics
#using Distributions
using StatsBase

println("test")

using Knet

println("test gpu")
println(Knet.gpuCount())
println(Knet.gpu())
Knet.gpu(0)
println(Knet.gpu())

include(pwd()*"/src/alpha stable dist/set_up.jl")
include(pwd()*"/src/abc algorithms/abc_rs.jl")
include(pwd()*"/src/generate training test data/generate_parameter_data_pairs.jl")

# load stored parameter data paris
#proposalas = Matrix(CSV.read("data/alpha stable/abc_data_parameters.csv", allowmissing=:auto))
#datasets = Matrix(CSV.read("data/alpha stable/abc_data_data.csv", allowmissing=:auto))

if Knet.gpuCount() > 0

    proposalas = Matrix(CSV.read("/lunarc/nobackup/users/samwiq/abc-dl/data/alpha stable/abc_data_parameters.csv", allowmissing=:auto))
    datasets = Matrix(CSV.read("/lunarc/nobackup/users/samwiq/abc-dl/data/alpha stable/abc_data_data.csv", allowmissing=:auto))

else

    proposalas = Matrix(CSV.read("data/alpha stable/abc_data_parameters.csv", allowmissing=:auto))
    datasets = Matrix(CSV.read("data/alpha stable/abc_data_data.csv", allowmissing=:auto))

end

data = zeros(size(datasets,1),size(datasets,2)+1)
data[:,1:size(datasets,2)] = datasets
data[:,end] = proposalas[:,3]

y_obs_temp = zeros(length(y_obs)+1)

y_obs_temp[1:end-1] = y_obs
y_obs_temp[end] = θ_true[3]

y_obs = y_obs_temp

# set functions to calc summary statistics
function calc_summary(y_sim::Vector, y_obs::Vector)

    # set data

    γ = y_sim[end]
    y_sim = y_sim[1:end-1]

    q_5 = quantile(y_sim, 0.05)
    q_25 = quantile(y_sim, 0.25)
    q_50 = quantile(y_sim, 0.5)
    q_75 = quantile(y_sim, 0.75)
    q_95 = quantile(y_sim, 0.95)

    v_alpha = (q_95-q_5)/(q_75-q_25)
    v_beta = (q_95+q_5-2*q_50)/(q_95-q_5)
    v_gamma = (q_75-q_25)/γ
    x_bar = mean(y_sim)

    return [v_alpha;v_beta;v_gamma;x_bar] #[v_alpha;v_beta;v_gamma;x_bar] #map_parameters([v_alpha;v_beta;v_gamma;x_bar])

end


calc_summary(y_obs,y_obs)


# compute weigths using a pre-run

w_pre_run  =  ones(4)
ρ_pre_run(s::Vector, s_star::Vector) = euclidean_dist(s::Vector, s_star::Vector, w_pre_run)


nbr_obs_pre_run = 100000

Random.seed!(111)
proposalas_pre_run, datasets_pre_run = generate_parameter_data_pairs(nbr_obs_pre_run, sample_from_prior, generate_data)

data_pre_run = zeros(size(datasets_pre_run,1),size(datasets_pre_run,2)+1)
data_pre_run[:,1:size(datasets_pre_run,2)] = datasets_pre_run
data_pre_run[:,end] = proposalas_pre_run[:,3]


approx_posterior_pre_run, datasets_posterior_pre_run  = @time abcrs(y_obs, proposalas_pre_run, data_pre_run, calc_summary, ρ_pre_run, return_data=true; cutoff_percentile=0.1)

summary_stats_pre_run = zeros(4, size(datasets_posterior_pre_run,1))

for i = 1:size(datasets_posterior_pre_run,1)
    summary_stats_pre_run[:,i] = calc_summary(datasets_posterior_pre_run[i,:], y_obs[:])
end

w  =  ones(4)

for i = 1:4; w[i] = mad(summary_stats_pre_run[i,:]; normalize=false); end

# create distance function
ρ(s::Vector, s_star::Vector) = euclidean_dist(s::Vector, s_star::Vector, w)


#=
for i = 1:500000
    map_parameters(proposalas[i,:])
end
=#

# run ABC-RS
approx_posterior_samples = @time abcrs(y_obs, proposalas[1:100000,:], data[1:100000,:], calc_summary, ρ, return_data=false; cutoff_percentile=5*0.02)

write_to_files = 0

if write_to_files == 1
    # save approx posterior samples
    CSV.write("data/alpha stable/abcrs_post.csv", DataFrame(approx_posterior_samples))
end
