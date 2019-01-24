# script to compare the training and predctions obtained from the different networks

using Pkg
using PyPlot
using DataFrames
using CSV

include(pwd()*"/src/MA2/set_up.jl")

y_test = Matrix(CSV.read("data/MA2/y_test.csv"; allowmissing=:auto))

# results for DNN simple

loss_training_dnn_simple = Matrix(CSV.read("data/MA2/loss_vec_training_DNN_simple.csv"; allowmissing=:auto))
loss_val_dnn_simple = Matrix(CSV.read("data/MA2/loss_vec_val_DNN_simple_1.csv"; allowmissing=:auto))
predictions_dnn_simple = Matrix(CSV.read("data/MA2/predictions_DNN_simple_1.csv"; allowmissing=:auto))

# plot training and val error
PyPlot.figure()
PyPlot.plot(1:length(loss_training_dnn_simple), loss_training_dnn_simple[:], "*-r")
PyPlot.plot(1:length(loss_training_dnn_simple), loss_val_dnn_simple[:], "*-b")

# plot predictions
PyPlot.figure()
PyPlot.plt[:hist2d](y_test[:,1], predictions_dnn_simple[:,1], bins=(100, 100))
PyPlot.figure()
PyPlot.plt[:hist2d](y_test[:,2], predictions_dnn_simple[:,2], bins=(100, 100))

PyPlot.figure()
PyPlot.plot(y_test[:,1], predictions_dnn_simple[:,1],"*")
PyPlot.figure()
PyPlot.plot(y_test[:,2],predictions_dnn_simple[:,2], "*")

# check pred error
loss_test = sqrt.(sum((predictions_dnn_simple - y_test).^2, dims=2))

PyPlot.figure()
h = PyPlot.plt[:hist](loss_test,100)


# results for CNN

loss_training_dnn_cnn = Matrix(CSV.read("data/MA2/loss_vec_training_cnn_1.csv"; allowmissing=:auto))
loss_val_dnn_cnn = Matrix(CSV.read("data/MA2/loss_vec_val_cnn_1.csv"; allowmissing=:auto))
predictions_dnn_cnn = Matrix(CSV.read("data/MA2/predictions_cnn_1.csv"; allowmissing=:auto))

# plot training and val error
PyPlot.figure()
PyPlot.plot(1:length(loss_training_dnn_cnn), loss_training_dnn_cnn[:], "*-r")
PyPlot.plot(1:length(loss_training_dnn_cnn), loss_val_dnn_cnn[:], "*-b")

# plot predictions
PyPlot.figure()
PyPlot.plt[:hist2d](y_test[:,1], predictions_dnn_cnn[:,1], bins=(100, 100))
PyPlot.figure()
PyPlot.plt[:hist2d](y_test[:,2], predictions_dnn_cnn[:,2], bins=(100, 100))

PyPlot.figure()
PyPlot.plot(y_test[:,1], predictions_dnn_cnn[:,1],  "*")
PyPlot.figure()
PyPlot.plot(y_test[:,2], predictions_dnn_cnn[:,2],  "*")

# check pred error
loss_test = sqrt.(sum((predictions_dnn_cnn - y_test).^2,dims = 2))

PyPlot.figure()
h = PyPlot.plt[:hist](loss_test,100)



# results for deepsets

loss_training_dnn_bp_deepsets = Matrix(CSV.read("data/MA2/loss_vec_training_bp_deepsets_1.csv"; allowmissing=:auto))
loss_val_dnn_bp_deepsets = Matrix(CSV.read("data/MA2/loss_vec_val_bp_deepsets_1.csv"; allowmissing=:auto))
predictions_dnn_bp_deepsets = Matrix(CSV.read("data/MA2/predictions_bp_deepsets_1.csv"; allowmissing=:auto))

# plot training and val error
PyPlot.figure()
PyPlot.plot(1:length(loss_training_dnn_bp_deepsets), loss_training_dnn_bp_deepsets[:], "*-r")
PyPlot.plot(1:length(loss_training_dnn_bp_deepsets), loss_val_dnn_bp_deepsets[:], "*-b")

# plot predictions
PyPlot.figure()
PyPlot.plt[:hist2d](y_test[:,1], predictions_dnn_bp_deepsets[:,1], bins=(100, 100))
PyPlot.figure()
PyPlot.plt[:hist2d](y_test[:,2], predictions_dnn_bp_deepsets[:,2], bins=(100, 100))

PyPlot.figure()
PyPlot.plot(y_test[:,1], predictions_dnn_bp_deepsets[:,1],  "*")
PyPlot.figure()
PyPlot.plot(y_test[:,2], predictions_dnn_bp_deepsets[:,2],  "*")

# check pred error
loss_test = sqrt.(sum((predictions_dnn_bp_deepsets - y_test).^2,dims = 2))

PyPlot.figure()
h = PyPlot.plt[:hist](loss_test,100)
