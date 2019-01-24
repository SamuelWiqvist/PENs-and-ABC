
using Statistics

# calc MSE loss
#loss(w,x,ygold) = mean(abs2,ygold-predict(w,x))
loss(w,x,ygold) = sum((ygold-predict(w,x)).^2)
loss_dropout(w,x,ygold,p) = sum((ygold-predict_dropout(w,x,p)).^2)

# calc gradient of loss
lossgradient = grad(loss)
lossgradient_dropout = grad(loss_dropout)

# update weigths
function train(model, data, optim)
    for (x,ygold) in data
        grads = lossgradient(model,x,ygold)
        update!(model, grads, optim)
    end
end

function train_load_batch_to_gpu(model, data, optim)

    for (x,ygold) in data

        if Knet.gpuCount() > 0
            x = convert(Knet.KnetArray, x)
            ygold = convert(Knet.KnetArray, ygold)
        end

        grads = lossgradient(model,x,ygold)
        update!(model, grads, optim)
    end
end

function train_load_batch_to_gpu_w_dropout(model, data, optim,p)

    for (x,ygold) in data

        if Knet.gpuCount() > 0
            x = convert(Knet.KnetArray, x)
            ygold = convert(Knet.KnetArray, ygold)
        end

        grads = lossgradient_dropout(model,x,ygold,p)
        update!(model, grads, optim)
    end
end
