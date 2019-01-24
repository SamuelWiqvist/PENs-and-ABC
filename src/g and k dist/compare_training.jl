# script to compare the training and predctions obtained from the different networks

using Pkg
using PyPlot
using DataFrames
using CSV

include(pwd()*"/src/g and k dist/set_up.jl")

y_test = Matrix(CSV.read("data/gandk/y_test.csv"; allowmissing=:auto))

# results for DNN simple small

loss_training_dnn_simple = Matrix(CSV.read("data/gandk/loss_vec_training_simple_DNN_small.csv"; allowmissing=:auto))
loss_val_dnn_simple = Matrix(CSV.read("data/gandk/loss_vec_val_simple_DNN_small.csv"; allowmissing=:auto))
predictions_dnn_simple = Matrix(CSV.read("data/gandk/predictions_simple_DNN_small.csv"; allowmissing=:auto))

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
PyPlot.plt[:hist2d](y_test[:,3], predictions_dnn_simple[:,3], bins=(100, 100))
PyPlot.figure()
PyPlot.plt[:hist2d](y_test[:,4], predictions_dnn_simple[:,4], bins=(100, 100))

# check pred error
loss_test = sqrt.(sum((predictions_dnn_simple - y_test).^2, dims=2))

PyPlot.figure()
h = PyPlot.plt[:hist](loss_test,100)


# results for DNN simple large

loss_training_dnn_large = Matrix(CSV.read("data/gandk/loss_vec_training_simple_DNN_large.csv"; allowmissing=:auto))
loss_val_dnn_large = Matrix(CSV.read("data/gandk/loss_vec_val_simple_DNN_large.csv"; allowmissing=:auto))
predictions_dnn_large = Matrix(CSV.read("data/gandk/predictions_simple_DNN_large.csv"; allowmissing=:auto))

# plot training and val error
PyPlot.figure()
PyPlot.plot(1:length(loss_training_dnn_large), loss_training_dnn_large[:], "*-r")
PyPlot.plot(1:length(loss_training_dnn_large), loss_val_dnn_large[:], "*-b")

# plot predictions
PyPlot.figure()
PyPlot.plt[:hist2d](y_test[:,1], predictions_dnn_large[:,1], bins=(100, 100))
PyPlot.figure()
PyPlot.plt[:hist2d](y_test[:,2], predictions_dnn_large[:,2], bins=(100, 100))
PyPlot.figure()
PyPlot.plt[:hist2d](y_test[:,3], predictions_dnn_large[:,3], bins=(100, 100))
PyPlot.figure()
PyPlot.plt[:hist2d](y_test[:,4], predictions_dnn_large[:,4], bins=(100, 100))

# check pred error
loss_test = sqrt.(sum((predictions_dnn_large - y_test).^2, dims=2))

PyPlot.figure()
h = PyPlot.plt[:hist](loss_test,100)


# results for DNN timeserirs

loss_training_dnn_timseries_data = Matrix(CSV.read("data/gandk/loss_vec_training_dnn_timseries_data.csv"; allowmissing=:auto))
loss_val_dnn_timseries_data = Matrix(CSV.read("data/gandk/loss_vec_val_dnn_timseries_data.csv"; allowmissing=:auto))
predictions_dnn_timseries_data = Matrix(CSV.read("data/gandk/predictions_dnn_timseries_data.csv"; allowmissing=:auto))

# plot training and val error
PyPlot.figure()
PyPlot.plot(1:length(loss_training_dnn_timseries_data), loss_training_dnn_timseries_data[:], "*-r")
PyPlot.plot(1:length(loss_training_dnn_timseries_data), loss_val_dnn_timseries_data[:], "*-b")

# plot predictions
PyPlot.figure()
PyPlot.plt[:hist2d](y_test[:,1], predictions_dnn_timseries_data[:,1], bins=(100, 100))
PyPlot.figure()
PyPlot.plt[:hist2d](y_test[:,2], predictions_dnn_timseries_data[:,2], bins=(100, 100))
PyPlot.figure()
PyPlot.plt[:hist2d](y_test[:,3], predictions_dnn_timseries_data[:,3], bins=(100, 100))
PyPlot.figure()
PyPlot.plt[:hist2d](y_test[:,4], predictions_dnn_timseries_data[:,4], bins=(100, 100))

# check pred error
loss_test = sqrt.(sum((predictions_dnn_timseries_data - y_test).^2, dims=2))

PyPlot.figure()
h = PyPlot.plt[:hist](loss_test,100)



# results for deepsets

loss_training_deepsets = Matrix(CSV.read("data/gandk/loss_vec_training_deepsets.csv"; allowmissing=:auto))
loss_val_deepsets = Matrix(CSV.read("data/gandk/loss_vec_val_deepsets.csv"; allowmissing=:auto))
predictions_deepsets = Matrix(CSV.read("data/gandk/predictions_deepsets.csv"; allowmissing=:auto))

# plot training and val error
PyPlot.figure()
PyPlot.plot(1:length(loss_training_deepsets), loss_training_deepsets[:], "*-r")
PyPlot.plot(1:length(loss_training_deepsets), loss_val_deepsets[:], "*-b")

# plot predictions
PyPlot.figure()
PyPlot.plt[:hist2d](y_test[:,1], predictions_deepsets[:,1], bins=(100, 100))
PyPlot.figure()
PyPlot.plt[:hist2d](y_test[:,2], predictions_deepsets[:,2], bins=(100, 100))
PyPlot.figure()
PyPlot.plt[:hist2d](y_test[:,3], predictions_deepsets[:,3], bins=(100, 100))
PyPlot.figure()
PyPlot.plt[:hist2d](y_test[:,4], predictions_deepsets[:,4], bins=(100, 100))

# check pred error
loss_test = sqrt.(sum((predictions_deepsets - y_test).^2, dims=2))

PyPlot.figure()
h = PyPlot.plt[:hist](loss_test,100)
