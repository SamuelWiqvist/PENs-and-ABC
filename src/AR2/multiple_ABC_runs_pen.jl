using Random

include(pwd()*"/src/AR2/train_pen.jl")

N = 100 # should be 100
samples = 500

posteriors = zeros(2*N,samples)

print("Round:")
println(1)

posteriors[1:2,:] = approx_posterior_samples'


for i = 3:2:(2*N-1)
    print("Round:")
    println(i)
    Random.seed!(i)

    y_obs = generate_data(θ_true)

    y_obs = restruct(y_obs,nbr_features,time_delay)[:]

    approx_posterior_samples = @time abcrs(y_obs, proposalas, datasets, calc_summary, ρ; cutoff_percentile=0.1)
    global posteriors[i:i+1,:] = approx_posterior_samples'
end

CSV.write("data/AR2/posteriors_multiple_data_sets_"*job*".csv", DataFrame(posteriors))
