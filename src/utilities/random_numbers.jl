
# generate uniform random varaibles
function rand_unif(a::Real, b::Real, n::Int)
    u = zeros(n)
    for i in 1:n; u[i] = a + (b-a)*rand(); end
    return u
end

# gamma random varaibles, using the method from https://www.cs.toronto.edu/~radford/csc2541.F04/gamma.html
function rand_gamma(α::Int, β::Real, n::Int=1)

    x = zeros(n)
    for i in 1:n
        u = rand(α)
        L = zeros(α)
        for i in 1:α; L[i] = -log(1-u[i]); end
        s = sum(L)
        x[i] = s/β
    end

    n == 1 ? (return x[1]) : return x

end

rand_exp(λ::Real) = -1/λ*log(1-rand())

#=
# test exponential

d = Exponential(1)
x_grid = -0.1:0.01:5

N = 100
s = zeros(N)
for i = 1:100; s[i] = rand_exp(1); end

# plot Exp cdf
PyPlot.figure()
PyPlot.plot(x_grid,cdf(d,x_grid), "k")
PyPlot.plot(x_grid, ecdf(s)(x_grid), "b")
=#
