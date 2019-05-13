# set up the AR2 process
using Random
using Statistics
using StatsBase

println("Loading AR2 model")

# Define model

# Sample from the Uniform prior distribution
function sample_from_prior()

  while true
    θ_1 = -2 + (2+2)*rand()
    θ_2 = -1 + (1+1)*rand()
    if validparameters(θ_1, θ_2)
      return [θ_1; θ_2]
    end
  end

end

# Evaluate prior distribtuion
function evaluate_prior(Θ::Vector)
  θ_1 = Θ[1]
  θ_2 = Θ[2]
  if validparameters(θ_1, θ_2)
    return 1.
  else
    return 0.
  end
end

# Generate data from the model
function generate_data(θ::Vector, N_data=100)

  θ_1 = θ[1]
  θ_2 = θ[2]

  if validparameters(θ_1,θ_2)

    y = zeros(N_data)
    ϵ = randn(N_data)

    y[1] = ϵ[1]
    y[2] = θ_1*y[1] + ϵ[2]

    @inbounds for i = 3:N_data
      y[i] = θ_1*y[i-1] + θ_2*y[i-2] + ϵ[i]
    end

    return y
  else
    return NaN*ones(N_data)
  end

end

# check if the parameters are valid, see https://stats.stackexchange.com/questions/118019/a-proof-for-the-stationarity-of-an-ar2
function validparameters(θ_1::Real,θ_2::Real)

  if θ_2 < 1 + θ_1 && θ_2 < 1 - θ_1 && θ_2 > -1
    return true
  else
    return false
  end

end


# ground-truth parameters
θ_true = [0.2; -0.13]

# load data
Random.seed!(1)
y_obs = generate_data(θ_true)

#=
using PyPlot
PyPlot.figure()
PyPlot.plot(y_obs)
PyPlot.xlabel("Time.")
=#
