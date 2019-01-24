# set up the g-ank-k dist

using DataFrames
using CSV
using Random
using StatsBase

include(pwd()*"/src/utilities/random_numbers.jl")

println("Loading alpha-stable model")

# Sample from prior dist
function sample_from_prior()

  α_tilde = randn()
  β_tilde = randn()
  γ_tilde = randn()
  δ_tilde = randn()

  return inverse_map_parameters([α_tilde;β_tilde;γ_tilde;δ_tilde])

end

function inverse_map_parameters(θ_tilde::Vector)

  α = (2*exp(θ_tilde[1]) + 1.1)/(1+exp(θ_tilde[1]))
  β = (exp(θ_tilde[2])-1)/(1+exp(θ_tilde[2]))
  γ = exp(θ_tilde[3])
  δ = θ_tilde[4]

  return [α;β;γ;δ]

end

function map_parameters(θ::Vector)

  α_tilde = log((θ[1]-1.1)/(2-θ[1]))
  β_tilde = log((θ[2]+1)/(1-θ[2]))
  γ_tilde = log(θ[3])
  δ_tilde = θ[4]

  return [α_tilde;β_tilde;γ_tilde;δ_tilde]

end


# Evaluate the prior dist on log-scale
function  evaluate_prior(Θ::Vector)

end

# Generate data from the model
function generate_data(θ::Vector; N_data::Int = 1000)

  #(α, β, γ, δ) = inverse_map_parameters(θ_tilde)
  (α, β, γ, δ) = θ

  samples = zeros(N_data)

  for i = 1:N_data; samples[i] = sample_uniform_alpha_stable(α,β,γ,δ); end

  return samples

end

function sample_uniform_alpha_stable(α::Real, β::Real, γ::Real, δ::Real)

   w = rand_exp(1)
   u = -pi/2 + pi*rand()

   y_bar = 0

   if α != 1
     S = (1+β^2*tan((pi*α)/2)^2)^(1/(2*α))
     B = (1/α)*atan(β*tan((pi*α)/2))
     y_bar = S*(sin(α*(u + B))/(cos(u)^(1/α)))*((cos(u-α*(u+B))/w)^((1-α)/α))
   else
     y_bar = (2/pi)*((pi/2 + β*u)*tan(u) - β*log(((pi/2)*w*cos(u))/((pi/2)+β*u)))
   end

   return γ*y_bar + δ
end

# ground-truth parameters
θ_true = [1.5; 0.5; 1; 0] # alpha mu_0 phi w

# load data
Random.seed!(123)
y_obs = generate_data(θ_true)

function remove_outliers(data::Array, low_cutoff::Real=-10, high_cutoff::Real=50)

    idx_low_values = findall(x->x<low_cutoff, data)
    idx_high_values = findall(x->x>high_cutoff, data)
    idx_domain = findall(x -> (x > low_cutoff && x < high_cutoff), data)

    if length(idx_low_values) > 0
      data[idx_low_values] = sample(data[idx_domain], length(idx_low_values), replace=true)
    end

    if length(idx_high_values) > 0
      data[idx_high_values] = sample(data[idx_domain], length(idx_high_values), replace=true)
    end

    return data
end



function robust_scaler(x::Array)

    for i = 1:size(x,2)
        x[i,:] = (x[i,:] .- quantile(x[i,:], 0.25))./(quantile(x[i,:], 0.75)-quantile(x[i,:], 0.25))
    end

end

function robust_scaler(x::Vector)
    return (x .- quantile(x, 0.25))./(quantile(x, 0.75)-quantile(x, 0.25))
end

function robust_scaler_return_quantiles(x::Vector)

    q1 = quantile(x, 0.25)
    q3 = quantile(x, 0.75)

    x_tilde = zeros(length(x)+2)

    x_tilde[1:end-2] = (x .- quantile(x, 0.25))./(quantile(x, 0.75)-quantile(x, 0.25))
    x_tilde[end-1] = q1
    x_tilde[end] = q3

    return x_tilde
end

#=
function minmax_scaler(x::Array)

    for i = 1:size(x,2)
        x[i,:] = (x[i,:] .- minimum(x[i,:]))./(maximum(x[i,:])-minimum(x[i,:]))
    end

end

function minmax_scaler(x::Vector)

    return (x .- minimum(x))./(maximum(x)-minimum(x))

end
=#
