# set up the g-ank-k dist

using DataFrames
using CSV
using Random
using StatsBase

include(pwd()*"/src/utilities/random_numbers.jl")

println("Loading g-and-k model")

θ_true = [3; 1; 2; .5]

# Sample from the Uniform prior distribution
sample_from_prior() = rand_unif(0, 10, 4)

# Sample from an informative prior dist
function sample_from_informative_prior()

    prior_A = rand_gamma(2,1)
    prior_B = rand_gamma(2,1)
    prior_g = rand_gamma(2,0.5)
    prior_k = rand_gamma(2,1)

    return [prior_A;prior_B;prior_g;prior_k]
end


# Evaluate the prior dist on log-scale
function  evaluate_prior(Θ::Vector)

  # set start value for loglik
  log_prior = 0.
  for i = 1:length(Θ)
    log_prior += log_unifpdf( Θ[i], 0, 10 )
  end

  return log_prior # return log_lik

end

# Generate data from the model
function generate_data(Θ::Vector, c::Real = 0.8, N_data::Int = 1000)

  A = Θ[1]
  B = Θ[2]
  g = Θ[3]
  k = Θ[4]

  z = randn(N_data)

  F_inv = similar(z)

  @inbounds for i = 1:length(F_inv)
    F_inv[i] = A + B*(1 + c*(1-exp(-g*z[i]))/(1+exp(-g*z[i])))*(1+z[i]^2)^k*z[i]
  end

  return F_inv

end

# Removes outliers from data, i.e. values that are below low_cutoff and above high_cutoff.
function remove_outliers(data::Array, low_cutoff::Real=-10, high_cutoff::Real=50)

    idx_low_values = findall(x->x<low_cutoff, data)
    idx_high_values = findall(x->x>high_cutoff, data)
    idx_domain = findall(x -> (x > low_cutoff && x < high_cutoff), data)

    data[idx_low_values] = sample(data[idx_domain], length(idx_low_values), replace=true)
    data[idx_high_values] = sample(data[idx_domain], length(idx_high_values), replace=true)

    return data
end
