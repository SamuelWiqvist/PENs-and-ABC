using Random

include(pwd()*"/src/MA2 noisy data/abcrs_inference.jl")

GC.gc()


N = 100
samples = 100

posteriors = zeros(2*N,samples)

print("Round:")
println(1)

posteriors[1:2,:] = approx_posterior_samples'

for i = 3:2:(2*N-1)
    print("Round:")
    println(i)
    Random.seed!(i)
    y_obs = convert(Array{Float32,1},generate_data(θ_true))
    approx_posterior_samples = @time abcrs(y_obs, proposalas, datasets, calc_summary, ρ; cutoff_percentile=0.02)
    global posteriors[i:i+1,:] = approx_posterior_samples'
end

#CSV.write("data/MA2 noisy data/posteriors_multiple_data_sets_abcrs.csv", DataFrame(posteriors))

CSV.write("/lunarc/nobackup/users/samwiq/abc-dl/data/MA2 noisy data/posteriors_multiple_data_sets_abcrs.csv", DataFrame(posteriors))
