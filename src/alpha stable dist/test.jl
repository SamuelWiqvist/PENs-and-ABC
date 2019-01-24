using PyPlot
using KernelDensity
using Distributions

include(pwd()*"/src/alpha stable dist/set_up.jl")



# test univariate alpha stable
# known alpha-stable dists http://math.bu.edu/people/mveillet/html/alphastablepub.html

# parameters for N(0,2)
α = 2
β = 0
γ = sqrt(2)
δ = 0
normal = Normal(0,2)
x_grid = -10:0.01:10

# parameters for Cauchy(1,1)
α = 1
β = 0
γ = 1
δ = 1
cauchy = Cauchy(1,1)
x_grid = -5:0.01:5

# parameters for Levy(1,1)
α = 0.5
β = 1
γ = 1
δ = 1
levy = Levy(1,1)
x_grid = -.1:0.01:100

N = 1000
x = zeros(N)

for i in 1:N; x[i] = sample_uniform_alpha_stable(α, β, γ, δ); end


PyPlot.figure()
h = PyPlot.plt[:hist](x,50)

# plot normal cdf
PyPlot.figure()
PyPlot.plot(x_grid,cdf(normal,x_grid), "k")
PyPlot.plot(x_grid, ecdf(x)(x_grid), "b")

# plot cauchy cdf
PyPlot.figure()
PyPlot.plot(x_grid,cdf(cauchy,x_grid), "k")
PyPlot.plot(x_grid, ecdf(x)(x_grid), "b")

# plot levy cdf
PyPlot.figure()
PyPlot.plot(x_grid,cdf(levy,x_grid), "k")
PyPlot.plot(x_grid, ecdf(x)(x_grid), "b")


function normplot(x::Vector)

  x = sort(x)
  p_i = zeros(length(x))
  z_i = zeros(length(x))
  prob_i = zeros(length(x))


  n = length(x)

  a = 1/3

  for i = 1:n
    p_i[i] = (i-a)/(n-a+1)
    z_i[i] = quantile(Normal(0,1), p_i[i])
    prob_i[i] = cdf(Normal(0,1), z_i[i])
  end

  PyPlot.figure(figsize=(7,5))
  PyPlot.plot(x, z_i, "b*")
  PyPlot.xlabel("Data")
  PyPlot.ylabel("Quantile")

  #=
  PyPlot.figure()
  PyPlot.plot(x, prob_i, "b*")
  PyPlot.xlabel("Data")
  PyPlot.ylabel("Probability")
  PyPlot.yscale("log")
  =#

end

normplot(x)

# test multivaraite alpha stable at true parameter values
Random.seed!(123)
y = generate_data(θ_true, N_data = 1000)

PyPlot.figure()
PyPlot.plot(y[1,:],y[2,:], "*")

y_kde = kde((y[1,:], y[2,:]))

PyPlot.figure()
PyPlot.plt[:contour](y_kde.x, y_kde.y, y_kde.density)

PyPlot.figure()
h = PyPlot.plt[:hist](y[1,:],100)

PyPlot.figure()
h = PyPlot.plt[:hist](y[2,:],100)


using KernelDensity

h1 = kde(y[1,:])
h2 = kde(y[2,:])

PyPlot.figure()
PyPlot.plot(h1.x,h1.density)

PyPlot.figure()
PyPlot.plot(h2.x,h2.density)

# test multivaraite alpha stable at random parameter values
cos(0)
cos(pi/2)


sin(pi/2)

θ = sample_from_prior()

# top non-symmertic
θ = [0.75;0;0;pi/2;1;1]
y = generate_data(θ, N_data = 1000)

# mid non-symmertic
θ = [1;0;0;pi/4;0.5;0.5]

Random.seed!(123)
y = generate_data(θ, N_data = 200)

PyPlot.figure()
PyPlot.plot(y[1,:],y[2,:], "*")

y_kde = kde((y[1,:], y[2,:]))

PyPlot.figure()
PyPlot.plt[:contour](y_kde.x, y_kde.y, y_kde.density)
