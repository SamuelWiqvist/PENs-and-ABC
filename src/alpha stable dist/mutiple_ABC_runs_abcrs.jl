using Random

write_to_files = 1

include(pwd()*"/src/alpha stable dist/abcrs_inference.jl")

GC.gc()

N = 25
samples = 100

posteriors = zeros(4*N,samples)

print("Round:")
println(1)

posteriors[1:4,:] = approx_posterior_samples'

@time for i = 5:4:(4*N-1)
    print("Round:")
    println(i)
    Random.seed!(i)
    y_obs = generate_data(θ_true)
    y_obs = [y_obs; θ_true[3]]
    approx_posterior_samples = @time abcrs(y_obs, proposalas[1:100000,:], datasets[1:100000,:], calc_summary, ρ, return_data=false; cutoff_percentile=5*0.02)
    global posteriors[i:i+3,:] = approx_posterior_samples'
end

if Knet.gpuCount() > 0

    CSV.write("/lunarc/nobackup/users/samwiq/abc-dl/data/alpha stable/posteriors_multiple_data_sets_abcrs.csv", DataFrame(posteriors))


else

    CSV.write("data/alpha stable/posteriors_multiple_data_sets_abcrs.csv", DataFrame(posteriors))

end
