# to run interactively set network and epoch, and commnet out the ARGS statments
# network = "DNN_simple" # DNN_simple or fully_conncted

println("start script")


network = ARGS[1] # DNN_simple/fully_conncted
training_alg = ARGS[2] # standard/early_stopping
epoch = parse(Int, ARGS[3]) # nbr epochs
data_size = parse(Int, ARGS[4]) # nbr training points
write_to_files = parse(Int, ARGS[5]) # nbr epochs

network = "DNN_simple"
training_alg = "standard"
epoch = 4
data_size = 4
write_to_files = 1


println(network)
println(training_alg)
println(epoch)
println(data_size)
println(write_to_files)

# load packages
using Pkg
using Printf

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

#println("test Knet")
#Pkg.test("Knet")

# load files
include(pwd()*"/src/MA2/set_up.jl")
include(pwd()*"/src/generate training test data/generate_parameter_data_pairs.jl")
include(pwd()*"/src/nets/generic_loss_grad_train.jl")
include(pwd()*"/src/nets/mlp_predict.jl")

# generate data
generate_training_test_data = false

if generate_training_test_data

    # generate parameter data paris
    nbr_obs_training = 10^6
    nbr_obs_val = div(nbr_obs_training,2)
    nbr_obs_test = 2*10^5

    nbr_obs = nbr_obs_training + nbr_obs_test + nbr_obs_val

    parameters, data = generate_parameter_data_pairs(nbr_obs, sample_from_prior, generate_data)

    # split parameter-data paris into test and training data

    X_training = data[1:nbr_obs_training,:]
    X_val = data[nbr_obs_training+1:nbr_obs_training+nbr_obs_val,:]
    X_test = data[nbr_obs_training+nbr_obs_val+1:end,:]

    y_training = parameters[1:nbr_obs_training,:]
    y_val = parameters[nbr_obs_training+1:nbr_obs_training+nbr_obs_val,:]
    y_test = parameters[nbr_obs_training+nbr_obs_val+1:end,:]


    CSV.write("data/MA2/X_training.csv", DataFrame(X_training))
    CSV.write("data/MA2/X_val.csv", DataFrame(X_val))
    CSV.write("data/MA2/X_test.csv", DataFrame(X_test))

    CSV.write("data/MA2/y_training.csv", DataFrame(y_training))
    CSV.write("data/MA2/y_val.csv", DataFrame(y_val))
    CSV.write("data/MA2/y_test.csv", DataFrame(y_test))

else

    # load training and test data
    X_training = Matrix(CSV.read("data/MA2/X_training.csv"; allowmissing=:auto))
    X_val = Matrix(CSV.read("data/MA2/X_val.csv"; allowmissing=:auto))
    X_test = Matrix(CSV.read("data/MA2/X_test.csv"; allowmissing=:auto))

    y_training = Matrix(CSV.read("data/MA2/y_training.csv"; allowmissing=:auto))
    y_val = Matrix(CSV.read("data/MA2/y_val.csv"; allowmissing=:auto))
    y_test = Matrix(CSV.read("data/MA2/y_test.csv"; allowmissing=:auto))

end

X_training = convert(Array{Float32,2},X_training')
X_val = convert(Array{Float32,2},X_val')
X_test = convert(Array{Float32,2},X_test')
y_training = convert(Array{Float32,2},y_training')
y_val = convert(Array{Float32,2},y_val')
y_test = convert(Array{Float32,2},y_test')
y_obs = convert(Array{Float32,1},y_obs)


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
# set up network
################################################################################

# set nbr features and outputs
nbr_features = size(X_training,1)
nbr_outputs = size(y_training,1)

# Network Parameters
if network == "DNN_simple"
    # simple network is the default setup
    n_hidden_1 = 100 # 1st layer number of neurons
    n_hidden_2 = 100 # 2nd layer number of neurons
    n_hidden_3 = 50 # 3nd layer number of neurons
    n_out = nbr_outputs # dim output
    n_input = nbr_features # dim input
elseif network == "fully_connected"
    n_hidden_1 = 80 # 1st layer number of neurons
    n_hidden_2 = 60 # 2nd layer number of neurons
    n_hidden_3 = 40 # 3nd layer number of neurons
    n_out = nbr_outputs # dim output
    n_input = nbr_features # dim input
end



# init weigths
# xavier initialization for relu activation functions

w = Any[ Float32(sqrt(2/n_input))*randn(Float32,n_hidden_1,n_input), zeros(Float32,n_hidden_1,1), # 0.1f0*randn(Float32,10,784) float32 numbers
         Float32(sqrt(2/n_hidden_1))*randn(Float32,n_hidden_2,n_hidden_1), zeros(Float32,n_hidden_2,1),
         Float32(sqrt(2/n_hidden_2))*randn(Float32,n_hidden_3,n_hidden_2), zeros(Float32,n_hidden_3,1),
         Float32(sqrt(2/n_hidden_3))*randn(Float32,n_out,n_hidden_3), zeros(Float32,n_out,1) ]

w_best = deepcopy(w)

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

# set optimizer
optimizer = optimizers(w, Adam)

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

y_pred = predict(w, X_test)


################################################################################
# save weigths, predictions and loss
################################################################################

job = network*"_"*string(data_size)


if write_to_files == 1

    CSV.write("data/MA2/loss_vec_training_"*job*".csv", DataFrame(loss_vec_training))
    CSV.write("data/MA2/loss_vec_val_"*job*".csv", DataFrame(loss_vec_val))
    CSV.write("data/MA2/predictions_"*job*".csv", DataFrame(y_pred'))

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
proposalas = Matrix(CSV.read("data/MA2/abc_data_parameters.csv", allowmissing=:auto))

datasets = Matrix(CSV.read("data/MA2/abc_data_data.csv", allowmissing=:auto))

proposalas = convert(Array{Float32,2},proposalas)
datasets = convert(Array{Float32,2},datasets)

# function to calc summary stats
calc_summary(y_sim::Vector, y_obs::Vector) = predict(w,y_sim)[:]

calc_summary(datasets[1,:], y_obs)

# distance function
ρ(s::Vector, s_star::Vector) = euclidean_dist(s, s_star, ones(Float32,2))

# run ABC-RS
approx_posterior_samples = @time abcrs(y_obs, proposalas, datasets, calc_summary, ρ; cutoff_percentile=0.02)

# save approx posterior samples
if write_to_files == 1
    CSV.write("data/MA2/"*job*"_abcrs_post.csv", DataFrame(approx_posterior_samples))
end

println("end script")
