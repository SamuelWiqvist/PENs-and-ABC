#import StatsBase.predict # we do not need to overwrite the predict function 

# calc predictions using a forward pass
function predict(w,x)
    #x = mat(x)
    for i=1:2:length(w)-2
        x = relu.(w[i]*x .+ w[i+1])
        #x = map(relu, x)
    end
    return w[end-1]*x .+ w[end]
end

function predict_dropout(w,x,p)
    #x = mat(x)
    for i=1:2:length(w)-2
        if i > 1
            x = dropout(x,p)
        end
        x = relu.(w[i]*x .+ w[i+1])
        #x = map(relu, x)
    end
    return w[end-1]*x .+ w[end]
end
