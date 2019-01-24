# posterior quantile intervals

# code form ApproximateBayesianComputation.jl

function quantile_interval(data::Matrix; lower::Real=2.5,upper::Real=97.5,  print_on::Bool = false)

    # find dim. for data
    dim = minimum(size(data))

    # set nbr of intervals
    intervals = zeros(dim, 2)

    # transform data to column-major order if necessary
    if size(data)[1] > size(data)[2]
        data = data'
    end

    # calc intervals over all dimensions
    for i = 1:dim
        intervals[i,:] = quantile(data[i,:], [lower/100 upper/100])
    end

    print_on == true ? show(intervals) :

    # return intervals
    return intervals

end


function quantile_interval(data::Vector; lower::Real=2.5,upper::Real=97.5, print_on::Bool = false)


    # set nbr of intervals
    intervals = zeros(1, 2)

    # calc intervals over all dimensions
    intervals[1,:] = quantile(data, [lower/100 upper/100])

    print_on == true ? show(intervals) :

    # return intervals
    return intervals

end

# loss

function loss(theta_true::Vector, theta_est::Vector)
  loss_vec = copy(theta_true)
  for i = 1:length(loss_vec)
    loss_vec[i] = abs(theta_true[i]-theta_est[i])
  end
  return loss_vec
end

function loss(theta_true::Real, theta_est::Real)
  return abs(theta_true-theta_est)
end
