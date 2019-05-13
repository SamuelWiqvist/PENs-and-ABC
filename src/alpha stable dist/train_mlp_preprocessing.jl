println("start script")

network = "mlp_preprocessing" # mlp/fully_conncted
training_alg = ARGS[1] # standard/early_stopping
epoch = parse(Int, ARGS[2]) # nbr epochs
n_hidden_1 = parse(Int,ARGS[3]) # 1st layer number of neurons
n_hidden_2 = parse(Int,ARGS[4]) # 2nd layer number of neurons
n_hidden_3 = parse(Int,ARGS[5]) # 3nd layer number of neurons
data_size = parse(Int, ARGS[6]) # nbr training points
write_to_file = parse(Int, ARGS[7]) # nbr training points

run_on_lunarc = true

#=
network = "dnn_timseries_data"
training_alg = "standard"
epoch = 2
n_hidden_1 = 100
n_hidden_2 = 100
n_hidden_3 = 50
data_size = 1
write_to_file = 1
=#

println(epoch)

# load packages
using Pkg
using Printf
using StatsBase

import StatsBase.sample

println("check version of Knet")
Pkg.status()

println("build Knet")
Pkg.build("Knet")

#println("test Knet")
#Pkg.test("Knet")

using Knet

println("test gpu")
println(Knet.gpuCount())
println(Knet.gpu())
Knet.gpu(0)
println(Knet.gpu())

# load files

include(pwd()*"/src/generate training test data/generate_parameter_data_pairs.jl")
include(pwd()*"/src/alpha stable dist/set_up.jl")
include(pwd()*"/src/nets/generic_loss_grad_train.jl")
include(pwd()*"/src/nets/mlp_predict.jl")

# load proposalas
if run_on_lunarc #Knet.gpuCount() > 0

    y_training = Matrix(CSV.read("/lunarc/nobackup/users/samwiq/abc-dl/data/alpha stable/y_training.csv"; allowmissing=:auto))
    y_val = Matrix(CSV.read("/lunarc/nobackup/users/samwiq/abc-dl/data/alpha stable/y_val.csv"; allowmissing=:auto))
    y_test = Matrix(CSV.read("/lunarc/nobackup/users/samwiq/abc-dl/data/alpha stable/y_test.csv"; allowmissing=:auto))


else

    y_training = Matrix(CSV.read("data/alpha stable/y_training.csv"; allowmissing=:auto))
    y_val = Matrix(CSV.read("data/alpha stable/y_val.csv"; allowmissing=:auto))
    y_test = Matrix(CSV.read("data/alpha stable/y_test.csv"; allowmissing=:auto))

end

# generate data
X_training = zeros(1000,size(y_training,1))
X_val = zeros(1000,size(y_val,1))
X_test  = zeros(1000,size(y_test,1))

Random.seed!(1235)
for i = 1:size(y_training,1)
    X_training[:,i] = generate_data(y_training[i,:])
end

Random.seed!(1236)
for i = 1:size(y_val,1)
    X_val[:,i] = generate_data(y_val[i,:])
end

Random.seed!(1237)
for i = 1:size(y_test,1)
    X_test[:,i] = generate_data(y_test[i,:])
end

GC.gc()

# convert to marginal emperical cdfs

cdf_lower = -10
cdf_upper = 100

x = LinRange(cdf_lower, cdf_upper, 100)

X_training_ecdf = zeros(100, size(X_training,2))
X_val_ecdf = zeros(100, size(X_val,2))
X_test_ecdf = zeros(100, size(X_test,2))

for i = 1:size(X_training,2)
    X_training_ecdf[:,i] = ecdf(X_training[:,i])(x)
end

for i = 1:size(X_val,2)
    X_val_ecdf[:,i] = ecdf(X_val[:,i])(x)
 end

 for i = 1:size(X_test,2)
     X_test_ecdf[:,i] = ecdf(X_test[:,i])(x)
 end

X_training = X_training_ecdf
X_val = X_val_ecdf
X_test = X_test_ecdf

# map paramters, i.e. the targets to the tilde parameterization
#=
for i = 1:size(y_training,1)
    y_training[i,:] = map_parameters(y_training[i,:])
end

for i = 1:size(y_val,1)
    y_val[i,:] = map_parameters(y_val[i,:])
end

for i = 1:size(y_test,1)
    y_test[i,:] = map_parameters(y_test[i,:])
end
=#

X_training = convert(Array{Float32,2},X_training)
X_val = convert(Array{Float32,2},X_val)
X_test = convert(Array{Float32,2},X_test)
y_training = convert(Array{Float32,2},y_training')
y_val = convert(Array{Float32,2},y_val')
y_test = convert(Array{Float32,2},y_test')
y_obs = convert(Array{Float32,1},y_obs)


GC.gc()

if data_size == 1
    # do nothing use full training data
elseif data_size == 2
    X_training = X_training[:,1:100000]
    y_training = y_training[:,1:100000]
elseif data_size == 3
    X_training = X_training[:,1:10000]
    y_training = y_training[:,1:10000]
elseif data_size == 4
    X_training = X_training[:,1:1000]
    y_training = y_training[:,1:1000]
end

################################################################################
## Set up Simple DNN model
################################################################################

# set nbr features and outputs
nbr_features = size(X_training,1)
nbr_outputs = size(y_training,1)

# Network Parameters
n_out = nbr_outputs # dim output
n_input = nbr_features # dim input

# init weigths
# xavier initialization for relu activation functions

w = Any[ Float32(sqrt(2/n_input))*randn(Float32,n_hidden_1,n_input), zeros(Float32,n_hidden_1,1),
         Float32(sqrt(2/n_hidden_1))*randn(Float32,n_hidden_2,n_hidden_1), zeros(Float32,n_hidden_2,1),
         Float32(sqrt(2/n_hidden_2))*randn(Float32,n_hidden_3,n_hidden_2), zeros(Float32,n_hidden_3,1),
         Float32(sqrt(2/n_hidden_3))*randn(Float32,n_out,n_hidden_3), zeros(Float32,n_out,1) ]

w_best = deepcopy(w)

# set optimizer
optimizer = optimizers(w, Adam)


nbr_training_obs = size(X_training,2)
nbr_parameters = 0

for i in w
    global nbr_parameters = nbr_parameters + size(i,1)*size(i,2)
end

network_info = @sprintf "Nbr training obs %d, nbr parameters %d, obs/parameters %.2f\n" nbr_training_obs nbr_parameters nbr_training_obs/nbr_parameters
print(network_info)

if Knet.gpuCount() > 0

    # set gpu
    Knet.gpu(0)
    println(Knet.gpu())

    # map to gpu
    w = map(KnetArray, w)
    #X_training = convert(Knet.KnetArray, X_training)
    X_val = convert(Knet.KnetArray, X_val)
    #X_test = convert(Knet.KnetArray, X_test)
    #y_training = convert(Knet.KnetArray, y_training)
    y_val = convert(Knet.KnetArray, y_val)
    #y_test = convert(Knet.KnetArray, y_test)

end

################################################################################
# training
################################################################################

# function for the training scheme
function training(epoch, batch_size)

    @printf "Starting training\n"

    best_val_loss = 0

    for i = 1:epoch

        # calc minibatches
        if Knet.gpuCount() > 0
            dtrn = minibatch(X_training, y_training, batch_size, shuffle=true, xtype=KnetArray)
        else
            dtrn = minibatch(X_training, y_training, batch_size, shuffle=true)
        end

        loss_sum = train_load_batch_to_gpu(w, dtrn, optimizer)

        idx_selected = sample(1:size(X_training,2), 5000, replace=true)

		if Knet.gpuCount() > 0

			# map to gpu
			X_temp = convert(Knet.KnetArray, X_training[:,idx_selected])
			y_temp = convert(Knet.KnetArray, y_training[:,idx_selected])

        else

            # map to gpu
            X_temp = X_training[:,idx_selected]
            y_temp = y_training[:,idx_selected]

        end

        loss_train = loss(w,X_temp, y_temp)/size(X_temp,2)
        # calc loss
        loss_val = loss(w, X_val, y_val)/size(X_val,2)

        loss_vec_training[i] = loss_train # store loss
        loss_vec_val[i] = loss_val

        if i == 1
            best_val_loss = loss_val
        elseif loss_val < best_val_loss
            best_val_loss = loss_val
            if Knet.gpuCount() > 0
                map_to_w_best!(w,w_best,Knet.gpuCount() > 0)
            else
                map_to_w_best!(w,w_best,Knet.gpuCount() > 0)
            end
        end

        # print current loss
        @printf "Epoch %d, current loss (training) %.4f, current loss (val) %.4f, best loss (val) %.4f \n" i loss_train loss_val best_val_loss

    end

end


# first training round using a small batch size
batch_size = 200

# pre-allocate vectors
loss_vec_training = zeros(epoch,1)
loss_vec_val = zeros(epoch,1)

# run training
if training_alg == "standard"
    run_time_first_training_cycle = @elapsed training(epoch, batch_size)
elseif training_alg == "early_stopping"
    error("early stopping is not implemented")
end


################################################################################
# Set weigths
################################################################################

w = w_best

if Knet.gpuCount() > 0

	w = map(Array, w)
	println(loss(map(KnetArray, w), X_val, y_val)/size(X_val,2))

else

	println(loss(w, X_val, y_val)/size(X_val,2))

end

################################################################################
# calc predictions
################################################################################

y_pred = predict(w, X_test)


################################################################################
# save weigths, predictions and loss
################################################################################

job = network*"_"*string(data_size)

if write_to_file == 1

    if run_on_lunarc #Knet.gpuCount() > 0

        CSV.write("/lunarc/nobackup/users/samwiq/abc-dl/data/alpha stable/loss_vec_training_"*job*".csv", DataFrame(loss_vec_training))
        CSV.write("/lunarc/nobackup/users/samwiq/abc-dl/data/alpha stable/loss_vec_val_"*job*".csv", DataFrame(loss_vec_val))
        CSV.write("/lunarc/nobackup/users/samwiq/abc-dl/data/alpha stable/predictions_"*job*".csv", DataFrame(y_pred'))


    else

        CSV.write("data/alpha stable/loss_vec_training_"*job*".csv", DataFrame(loss_vec_training))
        CSV.write("data/alpha stable/loss_vec_val_"*job*".csv", DataFrame(loss_vec_val))
        CSV.write("data/alpha stable/predictions_"*job*".csv", DataFrame(y_pred'))

    end

end

################################################################################
#  print runtime
################################################################################

s3 = @sprintf "run_time_first_training_cycle:  %.2f \n" run_time_first_training_cycle
s4 = @sprintf "Epochs %d, batch size, %d" epoch batch_size

println(s3)
println(s4)
println(network_info)


################################################################################
#  ABC inference
################################################################################

include(pwd()*"/src/abc algorithms/abc_rs.jl")

# load stored parameter data paris
if run_on_lunarc #Knet.gpuCount() > 0

    proposalas = Matrix(CSV.read("/lunarc/nobackup/users/samwiq/abc-dl/data/alpha stable/abc_data_parameters.csv", allowmissing=:auto))
    datasets = Matrix(CSV.read("/lunarc/nobackup/users/samwiq/abc-dl/data/alpha stable/abc_data_data.csv", allowmissing=:auto))

else

    proposalas = Matrix(CSV.read("data/alpha stable/abc_data_parameters.csv", allowmissing=:auto))
    datasets = Matrix(CSV.read("data/alpha stable/abc_data_data.csv", allowmissing=:auto))

end

y_obs = ecdf(y_obs)(x)

data_ecdf = zeros(size(datasets,1),100)

datasets

for i = 1:size(datasets,1)
     data_ecdf[i,:] = ecdf(datasets[i,:])(x)
 end

datasets = data_ecdf

# function to calc summary stats
calc_summary(y_sim::Vector, y_obs::Vector) = predict(w, y_sim)[:]

# distance function
ρ(s::Vector, s_star::Vector) = euclidean_dist(s, s_star, ones(4))

# run ABC-RS
approx_posterior_samples = abcrs(y_obs, proposalas[1:100000,:], datasets[1:100000,:], calc_summary, ρ, return_data=false; cutoff_percentile=5*0.02)

# save approx posterior samples

if write_to_file == 1

    if run_on_lunarc #Knet.gpuCount() > 0

        CSV.write("/lunarc/nobackup/users/samwiq/abc-dl/data/alpha stable/abcrs_post_"*job*".csv", DataFrame(approx_posterior_samples))

    else

        CSV.write("data/alpha stable/abcrs_post_"*job*".csv", DataFrame(approx_posterior_samples))

    end

end


println("end script")
