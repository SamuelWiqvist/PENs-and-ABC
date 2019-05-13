println("start script")


network = "pen2" # mlp/fully_conncted
training_alg = ARGS[1] # standard/early_stopping
epoch = parse(Int, ARGS[2]) # nbr epochs
data_size = parse(Int, ARGS[3]) # nbr training points
write_to_files = parse(Int, ARGS[4]) # nbr epochs

println(network)
println(training_alg)
println(epoch)
println(data_size)
println(write_to_files)

#=
network = "pen2"
training_alg = "standard"
epoch = 20
data_size = 1
write_to_files = 1
=#


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

#println("test Knet")
#Pkg.test("Knet")

using CSV
using DataFrames


# load files
include(pwd()*"/src/AR2/set_up.jl")
include(pwd()*"/src/generate training test data/generate_parameter_data_pairs.jl")
include(pwd()*"/src/nets/generic_loss_grad_train.jl")
include(pwd()*"/src/AR2/load_pen_data.jl")

################################################################################
## Set up BP-Deepsets model
################################################################################

time_delay = 2
generate_training_test_data = false


println("load data")

# generate data and transform data
X_training, X_val, X_test, y_training, y_val, y_test, proposalas, datasets, y_obs, nbr_features = @time load_data(y_obs,generate_training_test_data,data_size,time_delay)

# Network Parameters
n_input = time_delay+1 # dim input
n_out = size(y_training,1) # dim output

n_hidden_1 = 100
n_hidden_2 = 50
n_out_1 = 10

n_in_2 = 10+time_delay
n_hidden_3 = 50
n_hidden_4 = 50
n_hidden_5 = 20

# init weigths
# xavier initialization for relu activation functions

w = Any[ Float32(sqrt(2/n_input))*randn(Float32,n_hidden_1,n_input), zeros(Float32,n_hidden_1,1),
         Float32(sqrt(2/n_hidden_1))*randn(Float32,n_hidden_2,n_hidden_1), zeros(Float32,n_hidden_2,1),
         Float32(sqrt(2/n_hidden_2))*randn(Float32,n_out_1,n_hidden_2), zeros(Float32,n_out_1,1),

         Float32(sqrt(2/n_out_1))*randn(Float32,n_hidden_3,n_in_2), zeros(Float32,n_hidden_3,1),
         Float32(sqrt(2/n_hidden_3))*randn(Float32,n_hidden_4,n_hidden_3), zeros(Float32,n_hidden_4,1),
         Float32(sqrt(2/n_hidden_4))*randn(Float32,n_hidden_5,n_hidden_4), zeros(Float32,n_hidden_5,1),
         Float32(sqrt(2/n_hidden_5))*randn(Float32,n_out,n_hidden_5), zeros(Float32,n_out,1) ]

w_best = deepcopy(w)

function predict(w,x_input)

	# forward pass
    nbr_data_sets = size(x_input,2)
    x = reshape(x_input,time_delay+1,:)
    d_input = size(x,1)

    # inner network
    x = relu.(w[1]*x .+ w[2])
    x = relu.(w[3]*x .+ w[4])
    x = w[5]*x .+ w[6]

    # reshape and sum
    d_output = size(x,1)
    x = reshape(x, d_output, div(size(x,2),nbr_data_sets), nbr_data_sets)
    x = sum(x, dims = 2)./size(x,2)
    x = reshape(reshape(x, d_output,nbr_data_sets,1),d_output,nbr_data_sets)

    # add start values for the (assumed) markov model
    x = [x_input[1:time_delay,:]; x]

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

GC.gc()

#y_pred = predict(w, X_test, false)

y_pred = ones(2, size(X_test,2))

for i in 1:size(X_test,2); y_pred[:,i] = predict(w, X_test[:,i]); end

################################################################################
# save weigths, predictions and loss
################################################################################

job = network*"_"*string(data_size)

if write_to_files == 1
	CSV.write("data/AR2/loss_vec_training_"*job*".csv", DataFrame(loss_vec_training))
	CSV.write("data/AR2/loss_vec_val_"*job*".csv", DataFrame(loss_vec_val))
	CSV.write("data/AR2/predictions_"*job*".csv", DataFrame(y_pred'))
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
calc_summary(y_sim::Vector, y_obs::Vector) = predict(w,y_sim)[:]

calc_summary(datasets[1,:], y_obs)

# distance function
ρ(s::Vector, s_star::Vector) = euclidean_dist(s, s_star, ones(Float32,2))

# run ABC-RS

approx_posterior_samples = @time abcrs(y_obs, proposalas, datasets, calc_summary, ρ; cutoff_percentile=0.1)

# save approx posterior samples

if write_to_files == 1
	# save approx posterior samples
    CSV.write("data/AR2/"*job*"_abcrs_post.csv", DataFrame(approx_posterior_samples))
end

println("end script")
