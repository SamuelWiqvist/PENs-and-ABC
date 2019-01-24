using LinearAlgebra


# Compute multivariate Cramer statistics.
# Paper: https://www.sciencedirect.com/science/article/pii/S0047259X03000794?via%3Dihub
# R package for same test: https://cran.r-project.org/web/packages/cramer/index.html
function multvar_cramer_stat(x::Array,y::Array)

    m = size(x,2)
    n = size(y,2)

    s1 = 0
    s2 = 0
    s3 = 0

    for j in 1:m
        for k in 1:n
            s1 = s1 + kernel_FracA(norm(x[:,j] - y[:,k]))
        end
    end

    for j in 1:m
        for k in 1:m
            s2 = s2 + kernel_FracA(norm(x[:,j] - x[:,k]))
        end
    end


    for j in 1:n
        for k in 1:n
            s3 = s3 + kernel_FracA(norm(y[:,j] - y[:,k]))
        end
    end

    return m*n/(m+n)*(2/(m*n)*s1 - 1/m^2*s2-1/n^2*s3)

end


kernel_Bahr(x::Real) = 1-exp(-x/2)
kernel_FracA(x::Real) = 1-(1/(1+x))

#=
x = rand(MvNormal([0,0.],[1 .5;.5 1]),20)

y = rand(MvNormal([0,0.],[1 .5;.5 1]),20)

y = rand(MvNormal([10,10.],[1 .5;.5 1]),20)

y = rand(MvNormal([0,0.],[10 .5;.5 1]),20)


multvar_cramer_stat(x,y)
=#
