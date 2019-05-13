# set up the MA2 process
using DataFrames
using CSV

println("Loading MA2 noisy data model")

# ground-truth paraemters
θ_true = [0.6; 0.2]

# load data
ma2_data = Matrix(CSV.read("data/MA2 noisy data/ma2_data.csv"; allowmissing=:auto))
y_obs = ma2_data[:]

# Define model

# prior distribtuion

# The prior distribtuion: the parameters are defined on the triangle where
# -2 < θ_1 < 2, and  θ_1+θ_2 > 1, θ_1 - θ_2 < 1

# Sample from the Uniform prior distribution
function sample_from_prior()

  while true
    θ_1 = rand(Uniform(-2,2))
    θ_2 = rand(Uniform(-1,1))
    if θ_2 + θ_1  >= -1 && θ_2 - θ_1 >= -1
      return [θ_1; θ_2]
    end
  end

end

# Evaluate prior distribtuion
function evaluate_prior(Θ::Vector)
  θ_1 = Θ[1]
  θ_2 = Θ[2]
  if abs(θ_1) <= 2 && abs(θ_2) <= 1 && θ_2 + θ_1 >= -1 && θ_2 - θ_1 >= -1
    return 1.
  else
    return 0.
  end
end

# Generate data from the model
function generate_data(θ::Vector, N_data=100)

  θ_1 = θ[1]
  θ_2 = θ[2]

  if abs(θ_1) <= 2 && abs(θ_2) <= 1 && θ_2 + θ_1 >= -1 && θ_2 - θ_1 >= -1

    y = zeros(N_data)
    ϵ = randn(N_data)

    y[1] = ϵ[1]
    y[2] = ϵ[2] + θ_1*y[1]

    @inbounds for i = 3:N_data
      y[i] = ϵ[i] + θ_1*ϵ[i-1] + θ_2*ϵ[i-2]
    end

    σ_noise = 0.3

    y = y + σ_noise*randn(N_data)

    return y
  else
    return NaN*ones(N_data)
  end

end


# help functions

function calc_summary_stats_for_posterior(x)

    return [mean(x[1,:]),mean(x[2,:]), std(x[1,:]), std(x[2,:]), cor(x')[1,2]]

end

#=
using PyPlot

d = generate_data(θ_true)


d1 = generate_data(θ_true)

PyPlot.figure()
PyPlot.plot(d, "r")
PyPlot.plot(d1, "b")
PyPlot.xlabel("Time.")


data = zeros(1,100)

data[:] = d


CSV.write("data/MA2 noisy data/ma2_data.csv", DataFrame(data))
=#
