using Random

write_to_files = 0

include(pwd()*"/src/g and k dist/abcrs_inference.jl")

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
    y_obs = remove_outliers(y_obs, low_cutoff, high_cutoff)
    approx_posterior_samples = @time abcrs(y_obs, proposalas, datasets, calc_summary, ρ)
    global posteriors[i:i+3,:] = approx_posterior_samples'
end

CSV.write("data/gandk/posteriors_multiple_data_sets_abcrs.csv", DataFrame(posteriors))


