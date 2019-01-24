# load packages
#using PyPlot
using StatsBase

# load help function
include(pwd()*"/src/utilities/helpfunctions.jl")
include(pwd()*"/src/utilities/distancefunctions.jl")

# abc-rs
#Version of the ABC-RS algorithm where we return the proposalas with the
#'cutoff_percentile' lowest distances.
function abcrs(y_obs::Array, proposalas::Matrix, datasets::Matrix, calc_summary::Function, ρ::Function; cutoff_percentile::Real=0.1, return_data::Bool=false)

    @printf "Starting abc-rs\n"

    N = size(proposalas,1)
    ρ_vector = zeros(N)
    s = calc_summary(y_obs, y_obs)

    #@showprogress for i = 1:N
    for i = 1:N
        print_progress(i,N)
        s_star = calc_summary(datasets[i,:], y_obs)
        ρ_vector[i] = ρ(s, s_star)
    end

    ρ_val_cutoff = percentile(ρ_vector, cutoff_percentile)

    idx_keep = findall(x-> x <= ρ_val_cutoff, ρ_vector)

    @printf "Ending abc-rs\n"

    return_data ? (return (proposalas[idx_keep,:], datasets[idx_keep,:])) : return proposalas[idx_keep,:]

end
