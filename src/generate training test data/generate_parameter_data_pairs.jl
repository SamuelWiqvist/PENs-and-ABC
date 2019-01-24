# functions to generate the training data for the DL models
#using Distributions

# load help function
include(pwd()*"/src/utilities/helpfunctions.jl")

# Generate parameter data pairs

# Function to generate parameter-data pairs.
function generate_parameter_data_pairs(nbr_obs::Int,
                                        sample_from_prior::Function,
                                        generate_data::Function;
                                        print_interval::Int = 10^4)


    @printf "Starting: generate_parameter_data_pairs\n"

    # first iteration
    θ_star = sample_from_prior() # sample from prior and generate data
    y_star = generate_data(θ_star)

    # pre-allocate matricies
    data = zeros(nbr_obs, length(y_star))
    parameters = zeros(nbr_obs, length(θ_star))

    # store first data-parameter pair
    data[1,:] = y_star
    parameters[1,:] = θ_star

    for i = 2:nbr_obs

        print_progress(i,nbr_obs)

        θ_star = sample_from_prior() # sample from prior and generate data
        y_star = generate_data(θ_star)

        # store data-parameter pair
        data[i,:] = y_star
        parameters[i,:] = θ_star


    end

    return parameters, data

end
