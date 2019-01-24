# compute exact inference by numerically inverting the CDF, using the method
# by Dennis Prangel see https://arxiv.org/abs/1706.06889

# The code is addapted from https://github.com/SamuelWiqvist/introlikelihoodfree

using Roots # needed to inverte the quantile functon
using KernelDensity
using PyPlot
using Printf

# load gandk model
include(pwd()*"/src/g and k dist/set_up.jl")

################################################################################
## functions for defininf the g-and-k dist
################################################################################

"""
    Q(z::Real,A::Real,B::Real,g::Real, k::Real, c::Real)

Quantile function for the g-and-k distribution.
"""
function Q(z::Real,A::Real,B::Real,g::Real, k::Real, c::Real)

  return A + B*(1 + c*(1-exp(-g*z))/(1+exp(-g*z)))*(1+z^2)^k*z

end

"""
    Qprime(z::Real,A::Real,B::Real,g::Real, k::Real, c::Real)

Derivitie of the quantile function for the g-and-k distribution.
"""
function Qprime(z::Real,A::Real,B::Real,g::Real, k::Real, c::Real)

  return B*(1+z^2)^k*R(z,g,k)

end

"""
    R(z::Real,g::Real, k::Real)

Help function for the Q' function
"""
function R(z::Real,g::Real, k::Real)
  return (1+c*tanh(g*z/2))*((1+(2*k+1)*z^2)/(1+z^2))+((c*g*z)/(2*cosh(g*z/2)^2))
end


"""
    logQprime(z::Real,A::Real,B::Real,g::Real, k::Real, c::Real)

Logarithm of the derivitie of the quantile function for the g-and-k distribution.
"""
function logQprime(z::Real,A::Real,B::Real,g::Real, k::Real, c::Real)

  return log(B)+k*log(1+z^2)+logR(z,g,k)

end

"""
    logR(z::Real,g::Real, k::Real)

Help function for the logarithm of the Q' function
"""
function logR(z::Real,g::Real, k::Real)
  return log(R(z,g,k))
end


"""
    pdfgandk(x::Real, A::Real,B::Real,g::Real, k::Real, c::Real,logscale::Bool=true)

Numerical evaluating the pdf for the g-and-k distribution.
"""
function pdfgandk(x::Real, A::Real,B::Real,g::Real, k::Real, c::Real,logscale::Bool=true)


  # find z using rootfinder
  z = rootfinder(x,A,B,g,k,c)

  # calc pdf
  if !logscale
    return pdf(Normal(0,1), z)/Qprime(z,A,B,g,k,c)
  else
    return -0.5*log(2*pi) - 0.5*z^2  - logQprime(z,A,B,g,k,c)
  end

end


"""
    rootfinder(x::Real,A::Real,B::Real,g::Real, k::Real, c::Real)

Help function for pdfgandk.
"""
function rootfinder(x::Real,A::Real,B::Real,g::Real, k::Real, c::Real)

  f(z) = Q(z, A,B,g, k, c) - x
  z = fzero(f, -100, 100, no_pts=200)
  return z

end



"""
  pdfgandkdataset(y::Vector, A::Real,B::Real,g::Real, k::Real, c::Real,logscale::Bool=true)

Computes the (log)likelihood of the entire data set.
"""
function pdfgandkdataset(y::Vector, A::Real,B::Real,g::Real, k::Real, c::Real,logscale::Bool=true)

  if !logscale

    likelihood = 0.0

    for i = 1:length(y)
      likelihood = likelihood*pdfgandk(y[i], A,B,g, k, c,logscale)
    end

    return likelihood

  else

    loglikelihood = 0.0

    for i = 1:length(y)
      loglikelihood = loglikelihood + pdfgandk(y[i], A,B,g, k, c)
    end

    return loglikelihood

  end
end

################################################################################
# compare numerical pdf and compare with real pdf
################################################################################

h1 = kde(y_obs)

x = linspace(0,50, 100)

# set parameters
(A,B,g,k) = θ_true
c = 0.8

pdf_vec = zeros(length(x));

for i = 1:length(x)
  pdf_vec[i] = pdfgandk(x[i], A,B,g, k, c, false)
end


# plot simulated data
PyPlot.figure()
PyPlot.plot(h1.x,h1.density);
PyPlot.plot(x,pdf_vec, "r--");

################################################################################
# MH algorithm to run exact Bayesian inference
################################################################################


"""
    MH(y::Vector, nbr_iter::Int, adaptive_update, θ_0::Vector, c::Float64, print_interval::Int=100)

Runs Metropolis-Hastings algorithm.
"""
function MH(y::Vector, nbr_iter::Int, θ_0::Vector, conv_matrix::Array; c::Real=0.8, print_interval::Int=100)

  # pre-allocate matrices
  chain = zeros(length(θ_0), nbr_iter)
  loglik_vec = zeros(nbr_iter)
  accept_vec = zeros(nbr_iter)

  # set start values
  chain[:,1] = θ_0
  loglik_vec[1] = pdfgandkdataset(y,chain[1,1],chain[2,1],chain[3,1],chain[4,1],c)

  accept = true

  # print start
  @printf "Starting adaptive Metropolis-Hastings\n"



  for i = 2:nbr_iter

    # print info
    if mod(i-1,print_interval) == 0
      # print progress
      @printf "Percentage done: %.2f\n" 100*(i-1)/nbr_iter
      # print current acceptance rate
      @printf "Acceptance rate on iteration %d to %d is %.4f\n" i-print_interval i-1  sum(accept_vec[i-print_interval:i-1])/( i-1 - (i-print_interval) )
      # print loglik
      @printf "Loglik: %.4f \n" loglik_vec[i-1]

    end

    # random walk proposal
    # θ_proposal = chain[:,i-1] + rand(Normal(0,1),length(θ_0))*σ_mh

    # Gaussian random walk
    θ_proposal = rand(MvNormal(chain[:,i-1], conv_matrix))

    if log_prior(θ_proposal) == -Inf

      accept = false
      println("Proposal outside of domain of prior:")
      println(θ_proposal)

    else

      # loglik for proposal
      loglik_proposal = pdfgandkdataset(y,θ_proposal[1],θ_proposal[2],θ_proposal[3],θ_proposal[4],c)

      # compute logarithm of accaptance probability
      α_log = loglik_proposal + log_prior(θ_proposal) - (pdfgandkdataset(y,chain[1,i-1],chain[2,i-1],chain[3,i-1],chain[4,i-1],c) + log_prior(chain[:,i-1]))

      # generate log random number
      u_log = log(rand())

      # compute accaptance decision
      accept = u_log < α_log
    end

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

# help functions for adaptiveMH

function log_prior(θ::Vector)

  loglik_A = logpdf(Gamma(2,1), θ[1])
  loglik_B = logpdf(Gamma(2,1), θ[2])
  loglik_g = logpdf(Gamma(2,0.5), θ[3])
  loglik_k = logpdf(Gamma(2,1), θ[4])

  return loglik_A + loglik_B + loglik_g + loglik_k

end

function log_unifpdf(x::Real, a::Real, b::Real)

  if  x >= a && x<= b
    return -log(b-a)
  else
    return log(0)
  end

end

################################################################################
# calc exact inference
################################################################################

@time include(pwd()*"/src/g and k dist/generate_iid_data.jl")

using LinearAlgebra
using Random
using Distributions

nbr_iter = 15000 # nbr iterations
burn_in = 5000 # burn-in
θ_0 =  [5;5;3; 2.] # start values
conv_matrix = Array(Diagonal([0.002;0.002;0.002;0.002])) # cov matrix for RW proposal

# fix random numbers
Random.seed!(100)

chain, loglik_vec, accept_vec = @time MH(y_obs, nbr_iter,θ_0,conv_matrix; print_interval=25000)

# plotting

PyPlot.figure()
PyPlot.subplot(221)
PyPlot.plot(chain[1,:])
PyPlot.plot(ones(size(chain,2),1)*θ_true[1], "k")
PyPlot.ylabel(L"$A$")
PyPlot.subplot(222)
PyPlot.plot(chain[2,:])
PyPlot.plot(ones(size(chain,2),1)*θ_true[2], "k")
PyPlot.ylabel(L"$B$")
PyPlot.subplot(223)
PyPlot.plot(chain[3,:])
PyPlot.plot(ones(size(chain,2),1)*θ_true[3], "k")
PyPlot.ylabel(L"$g$")
PyPlot.xlabel(L"Iteration")
PyPlot.subplot(224)
PyPlot.plot(chain[4,:])
PyPlot.plot(ones(size(chain,2),1)*θ_true[4], "k")
PyPlot.xlabel(L"Iteration")
PyPlot.ylabel(L"$k$")

PyPlot.figure()
PyPlot.plot(loglik_vec)


# calc grid for prior dist
x_grid = -0.5:0.01:10.5

# calc prior dist
priordensity1 = pdf.(Gamma(2,1), x_grid)
priordensity2 = pdf.(Gamma(2,1), x_grid)
priordensity3 = pdf.(Gamma(2,0.5), x_grid)
priordensity4 = pdf.(Gamma(2,1), x_grid)


h1 = kde(chain[1,burn_in:end])
h2 = kde(chain[2,burn_in:end])
h3 = kde(chain[3,burn_in:end])
h4 = kde(chain[4,burn_in:end])

PyPlot.figure()
subplot(221)
PyPlot.plot(h1.x,h1.density, "b")
PyPlot.plot(x_grid,priordensity1, "g")
PyPlot.plot((θ_true[1], θ_true[1]), (0, maximum(h1.density)), "k")
PyPlot.ylabel(L"Density")
PyPlot.xlabel(L"$A$")
subplot(222)
PyPlot.plot(h2.x,h2.density, "b")
PyPlot.plot(x_grid,priordensity2, "g")
PyPlot.plot((θ_true[2], θ_true[2]), (0, maximum(h2.density)), "k")
PyPlot.xlabel(L"$B$")
subplot(223)
PyPlot.plot(h3.x,h3.density, "b")
PyPlot.plot(x_grid,priordensity3, "g")
PyPlot.plot((θ_true[3], θ_true[3]), (0, maximum(h3.density)), "k")
PyPlot.xlabel(L"$g$")
PyPlot.ylabel(L"Density")
subplot(224)
PyPlot.plot(h4.x,h4.density, "b")
PyPlot.plot(x_grid,priordensity4, "g")
PyPlot.plot((θ_true[4], θ_true[4]), (0, maximum(h4.density)), "k")
PyPlot.xlabel(L"$k$")


# save approx posterior samples
CSV.write("data/gandk/exact_inference_post.csv", DataFrame(chain[:,burn_in:end]'))

# exact inference for multiple datasets
using Random

N = 100
samples = size(chain[:,burn_in:end],2)

posteriors = zeros(4*N,samples)

print("Round:")
println(1)

posteriors[1:4,:] = chain[:,burn_in:end]

for i = 5:4:(4*N-1)
    print("Round:")
    println(i)
    Random.seed!(i)
    y_obs = generate_data(θ_true)
    y_obs = remove_outliers(y_obs, low_cutoff, high_cutoff)
    chain, loglik_vec, accept_vec = @time MH(y_obs, nbr_iter,θ_0,conv_matrix; print_interval=25000)
    global posteriors[i:i+3,:] = chain[:,burn_in:end]
end

CSV.write("data/gandk/posteriors_multiple_data_sets_exact.csv", DataFrame(posteriors'))
