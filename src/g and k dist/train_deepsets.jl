println("start script")


network = "deepsets" # DNN_simple/fully_conncted
training_alg = ARGS[1] # standard/early_stopping
epoch = parse(Int, ARGS[2])
data_size = parse(Int, ARGS[3])
write_to_file = parse(Int, ARGS[4])

#=
network = "deepsets"
training_alg = "standard"
epoch = 5
data_size = 4
write_to_file = 1
=#

println(epoch)

# load packages
using Pkg
using Printf

import StatsBase.sample
#import StatsBase.predict

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
include(pwd()*"/src/g and k dist/set_up.jl")
include(pwd()*"/src/nets/generic_loss_grad_train.jl")

#generate_training_test_data = false

@time include(pwd()*"/src/g and k dist/generate_iid_data.jl")

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

n_input = 1
n_hidden_1 = 100
n_hidden_2 = 50
n_out_1 = 10

n_hidden_3 = 100
n_hidden_4 = 100
n_hidden_5 = 50


# init weigths
# xavier initialization for relu activation functions

w = Any[ Float32(sqrt(2/n_input))*randn(Float32,n_hidden_1,n_input), zeros(Float32,n_hidden_1,1),
         Float32(sqrt(2/n_hidden_1))*randn(Float32,n_hidden_2,n_hidden_1), zeros(Float32,n_hidden_2,1),
         Float32(sqrt(2/n_hidden_2))*randn(Float32,n_out_1,n_hidden_2), zeros(Float32,n_out_1,1),

         Float32(sqrt(2/n_out_1))*randn(Float32,n_hidden_3,n_out_1), zeros(Float32,n_hidden_3,1),
         Float32(sqrt(2/n_hidden_3))*randn(Float32,n_hidden_4,n_hidden_3), zeros(Float32,n_hidden_4,1),
         Float32(sqrt(2/n_hidden_4))*randn(Float32,n_hidden_5,n_hidden_4), zeros(Float32,n_hidden_5,1),
         Float32(sqrt(2/n_hidden_5))*randn(Float32,n_out,n_hidden_5), zeros(Float32,n_out,1) ]

w_best = deepcopy(w)

function predict(w,x_input)

	# forward pass
    x = x_input
    d_input = size(x,1)
    nbr_datasets = size(x,2)

    x = x[:] # flat

    x = reshape(x, 1, length(x)) # reshape

    # inner network
    x = relu.(w[1]*x .+ w[2])
    x = relu.(w[3]*x .+ w[4])
    x = w[5]*x .+ w[6]

    # reshape and sum
    d_output = size(x,1)
    x = reshape(x, d_output, d_input, nbr_datasets)
    x = sum(x, dims = 2)./size(x,2)
    x = reshape(reshape(x, d_output,nbr_datasets,1),d_output,nbr_datasets)

    # outer network
    x = relu.(w[7]*x .+ w[8])
    x = relu.(w[9]*x .+ w[10])
    x = relu.(w[11]*x .+ w[12])
    x = w[13]*x .+ w[14]

	return x

end


# set optimizer
optimizer = optimizers(w, Adam)

nbr_training_obs = size(X_training,2)
nbr_parameters = 0


for i in w
    global nbr_parameters = nbr_parameters + size(i,1)*size(i,2)
end

network_info = @sprintf "Nbr training obs %d, nbr parameters %d, obs/parameters %.2f\n" nbr_training_obs nbr_parameters nbr_training_obs/nbr_parameters
print(network_info)

################################################################################
# training
################################################################################

# function for the training scheme
function training(epoch, batch_size)

    @printf "Starting training\n"

    best_val_loss = 0

    for i = 1:epoch

        # calc minibatches
        dtrn = minibatch(X_training, y_training, batch_size, shuffle=true)

        # update weigths
        train_load_batch_to_gpu(w, dtrn, optimizer)

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

        loss_vec_training[i] = loss_train
        loss_vec_val[i] = loss_val# store loss

        if i == 1
            best_val_loss = loss_val
        elseif loss_val < best_val_loss
            best_val_loss = loss_val
            if Knet.gpuCount() > 0
                w_best = deepcopy(map(Array, w))
            else
                w_best = deepcopy(w)
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

################################################################################
# calc predictions
################################################################################

GC.gc()

#y_pred = predict(w, X_test, false)
#        grads = lossgradient(model,x,ygold)

y_pred = ones(4, size(X_test,2))

for i in 1:size(X_test,2); y_pred[:,i] = predict(w, X_test[:,i]); end


################################################################################
# save weigths, predictions and loss
################################################################################

job = network*"_"*string(data_size)

if write_to_file == 1
    CSV.write("data/gandk/loss_vec_training_"*job*".csv", DataFrame(loss_vec_training))
    CSV.write("data/gandk/loss_vec_val_"*job*".csv", DataFrame(loss_vec_val))
    CSV.write("data/gandk/predictions_"*job*".csv", DataFrame(y_pred'))
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

# function to calc summary stats
calc_summary(y_sim::Vector, y_obs::Vector) = predict(w, y_sim)[:]

# distance function
ρ(s::Vector, s_star::Vector) = euclidean_dist(s, s_star, ones(4))

# run ABC-RS
approx_posterior_samples = abcrs(y_obs, proposalas, datasets, calc_summary, ρ)

# save approx posterior samples
if write_to_file == 1
    CSV.write("data/gandk/abcrs_post_"*job*".csv", DataFrame(approx_posterior_samples))
end

println("end script")
