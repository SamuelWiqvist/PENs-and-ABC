using Random

include(pwd()*"/src/alpha stable dist/train_deepsets.jl")

GC.gc()

N = 100 # should be 100 
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
    y_obs_input = robust_scaler_return_quantiles(remove_outliers(y_obs))
    approx_posterior_samples = @time abcrs(y_obs_input, proposalas[1:100000,:], datasets_input[1:100000,:], calc_summary, ρ, return_data=false; cutoff_percentile=0.02*5)
    global posteriors[i:i+3,:] = approx_posterior_samples'
end

CSV.write("/lunarc/nobackup/users/samwiq/abc-dl/data/alpha stable/posteriors_multiple_data_sets_"*job*".csv", DataFrame(posteriors))
