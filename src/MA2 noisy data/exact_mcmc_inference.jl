
using Distributions
using StatsBase
using KernelDensity
using PyPlot
using CSV
using Printf

include(pwd()*"/src/MA2 noisy data/set_up.jl")

# loglik function
function loglik(y,θ_proposal)

  # calc loglik using the method at: # http://economia.unipv.it/pagp/pagine_personali/erossi/rossi_ARMA_estimation_PhD.pdf

  ϵ = zeros(length(y))
  ϵ[1] = y[1]
  ϵ[2] = y[2] - θ_proposal[1]*ϵ[1]

  for i = 3:length(y)
    ϵ[i] = y[i] - θ_proposal[1]*ϵ[i-1] - θ_proposal[2]*ϵ[i-2]
  end

  return - sum(ϵ.^2)

end

# simple MHRW algorithm
function mh(y::Vector, nbr_iter::Int, σ_mh::Real, θ_0::Vector, print_interval::Int=500)

  # pre-allocate matrices
  chain = zeros(length(θ_0), nbr_iter)
  loglik_vec = zeros(nbr_iter)
  accept_vec = zeros(nbr_iter)

  # set start values
  chain[:,1] = θ_0
  loglik_vec[1] = loglik(y,chain[:,1])

  # print start
  @printf "Starting Metropolis-Hastings\n"

  for i = 2:nbr_iter

    # print info
    if mod(i-1,print_interval) == 0
      # print progress
      @printf "Percentage done: %.2f\n" 100*(i-1)/nbr_iter
      # print current acceptance rate
      @printf "Acceptance rate on iteration %d to %d is %.4f\n" i-print_interval i-1  sum(accept_vec[i-print_interval:i-1])/( i-1 - (i-print_interval) )
    end

    # random walk proposal
    θ_proposal = chain[:,i-1] + rand(Normal(0,1),length(θ_0))*σ_mh

    loglik_proposal = loglik(y,θ_proposal)
    # compute logarithm of accaptance probability
    α_log = loglik_proposal + log(evaluate_prior(θ_proposal)) - (loglik(y,chain[:,i-1]) + log(evaluate_prior(chain[:,i-1])))

    # generate log random number
    u_log = log(rand())

    # compute accaptance decision
    accept = u_log < α_log

    # update chain
    if accept
      chain[:,i] = θ_proposal
      loglik_vec[i] = loglik_proposal
      accept_vec[i] = 1
    else
      chain[:,i] = chain[:,i-1]
      loglik_vec[i] = loglik_vec[i-1]
    end

  end

  # print info
  @printf "Ending Metropolis-Hastings\n"
  return chain, loglik_vec, accept_vec

end

# burn-in
burn_in = 5000

# run RWMH
chain, loglik_vec, accept_vec = @time mh(y_obs, 105000, 0.17, [0;0])

#=
# Plot results
PyPlot.figure()
PyPlot.plot(chain[1,:])

PyPlot.figure()
PyPlot.plot(chain[2,:])

PyPlot.figure()
PyPlot.plot(chain[1,burn_in:end])
PyPlot.plot(θ_true[1]*ones(size(chain[1,burn_in:end],1)), "k")

PyPlot.figure()
PyPlot.plot(chain[2,burn_in:end])
PyPlot.plot(θ_true[2]*ones(size(chain[1,burn_in:end],1)), "k")

PyPlot.figure()
PyPlot.plot(loglik_vec)

PyPlot.figure()
PyPlot.plot((0,-2),(-1,1), "g")
PyPlot.plot((-2,2),(1,1), "g")
PyPlot.plot((0,2),(-1,1), "g")
PyPlot.scatter(chain[1,burn_in:end],chain[2,burn_in:end])
PyPlot.plot(θ_true[1],θ_true[2], "k*")
PyPlot.xlabel(L"$\theta_1$")
PyPlot.ylabel(L"$\theta_2$")
=#

# save samples from the posterior
CSV.write("data/MA2 noisy data/exact_mcmc_post.csv", DataFrame(chain[:,burn_in:end]'))

# exact inference for multiple datasets
using Random


N = 100
samples = size(chain[:,burn_in:end],2)

posteriors = zeros(2*N,samples)

print("Round:")
println(1)

posteriors[1:2,:] = chain[:,burn_in:end]

for i = 3:2:(2*N-1)
    print("Round:")
    println(i)
    Random.seed!(i)
    y_obs = convert(Array{Float32,1},generate_data(θ_true))
    chain, loglik_vec, accept_vec = mh(y_obs, 105000, 0.17, [0;0])
    global posteriors[i:i+1,:] = chain[:,burn_in:end]
end

CSV.write("data/MA2 noisy data/posteriors_multiple_data_sets_exact.csv", DataFrame(posteriors'))
