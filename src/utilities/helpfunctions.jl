# function to print progress of various algorithms
using Printf
using LinearAlgebra

function print_progress(i::Int, N::Int, print_interval::Int=10000)
    if mod(i,print_interval) == 0
        @printf("Percentage done:  %.2f %%\n", i/N*100)
    end
end

function log_normalpdf(μ::Real, σ::Real, x::Real)

  R = σ^2
  ϵ = x-μ
  return -0.5*(log(det(R)) + ϵ*inv(R)*ϵ)

end
