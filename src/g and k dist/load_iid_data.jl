using Pkg
using Random
using StatsBase

# load files iid data
include(pwd()*"/src/g and k dist/set_up.jl")

# load data
y_obs = CSV.read("data/gandk/y_obs.csv"; allowmissing=:auto)[:,2]
y_obs = convert(Array{Float32,1},y_obs)

# load ABC data
proposalas = convert(Array{Float32,2},CSV.read("data/gandk/ABC_proposals.csv", allowmissing=:auto))
datasets = convert(Array{Float32,2},CSV.read("data/gandk/ABC_datasets.csv", allowmissing=:auto))

# load training data for DNN
X_training = zeros(Float32, 1000, 100000,10)
X_val = zeros(Float32,1000, 100000)
X_test = zeros(Float32,1000,100000,2)

y_training = zeros(Float32,4,100000,10)
y_val = zeros(Float32,4,100000)
y_test = zeros(Float32,4,100000,2)

println("Load training data:")
for i in 1:10
    println(i)
    X_training[:,:,i] = convert(Array{Float32,2}, CSV.read("data/gandk/X_training"*string(i)*".csv"; allowmissing=:auto))'
    y_training[:,:,i] = convert(Array{Float32,2}, CSV.read("data/gandk/y_training"*string(i)*".csv"; allowmissing=:auto))'
end

println("Load test data:")
for i in 1:2
    println(i)
    X_test[:,:,i] = convert(Array{Float32,2}, CSV.read("data/gandk/X_test"*string(i)*".csv"; allowmissing=:auto))'
    y_test[:,:,i] = convert(Array{Float32,2}, CSV.read("data/gandk/y_test"*string(i)*".csv"; allowmissing=:auto))'
end

X_val = convert(Array{Float32,2}, CSV.read("data/gandk/X_val.csv"; allowmissing=:auto))'
y_val = convert(Array{Float32,2}, CSV.read("data/gandk/y_val.csv"; allowmissing=:auto))'

X_training = reshape(X_training, (1000,10*100000))
y_training = reshape(y_training, (4,10*100000))

X_test = reshape(X_test, (1000,2*100000))
y_test = reshape(y_test, (4,2*100000))
