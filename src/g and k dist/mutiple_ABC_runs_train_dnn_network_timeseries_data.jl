using Random

include(pwd()*"/src/g and k dist/train_simple_dnn_network_timeseries_data.jl")

GC.gc()

N = 100
samples = 100

posteriors = zeros(4*N,samples)

print("Round:")
println(1)

posteriors[1:4,:] = approx_posterior_samples'

for i = 5:4:(4*N-1)
    print("Round:")
    println(i)
    Random.seed!(i)
    y_obs = generate_data(θ_true)
    y_obs = ecdf(y_obs)(x)
    approx_posterior_samples = @time abcrs(y_obs, proposalas, datasets, calc_summary, ρ)
    global posteriors[i:i+3,:] = approx_posterior_samples'
end

CSV.write("data/gandk/posteriors_multiple_data_sets_"*job*".csv", DataFrame(posteriors))
