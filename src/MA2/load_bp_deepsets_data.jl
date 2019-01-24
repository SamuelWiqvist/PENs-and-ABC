function load_data(y_obs,generate_training_test_data,data_size,time_delay)

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


    nbr_features = size(X_training,1)
    nbr_outputs = size(y_training,1)

    X_training = restruct(X_training,nbr_features,time_delay)
    X_val = restruct(X_val,nbr_features,time_delay)
    X_test = restruct(X_test,nbr_features,time_delay)


    # load stored parameter data paris
    proposalas = Matrix(CSV.read("data/MA2/abc_data_parameters.csv", allowmissing=:auto))

    datasets = Matrix(CSV.read("data/MA2/abc_data_data.csv", allowmissing=:auto))

    proposalas = convert(Array{Float32,2},proposalas)
    datasets = Array(convert(Array{Float32,2},datasets)')

    datasets = Array(restruct(datasets,nbr_features,time_delay)')
    y_obs = restruct(y_obs,nbr_features,time_delay)[:]


    return X_training, X_val, X_test, y_training, y_val, y_test, proposalas, datasets, y_obs, nbr_features

end




# restruct data
function restruct(X::Array,nbr_features::Int,time_delay::Int)

    X_restruct = zeros(Float32,(nbr_features-time_delay)*(time_delay+1), size(X,2))

    idx_start_restruct = 1

    for i in 1:(size(X,1)-time_delay)
    	X_restruct[idx_start_restruct:idx_start_restruct+time_delay,:] = X[i:i+time_delay,:]
        idx_start_restruct = idx_start_restruct + (time_delay+1)
	end

    return X_restruct

end
