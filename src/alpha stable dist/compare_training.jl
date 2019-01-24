# script to compare the training and predctions obtained from the different networks

using Pkg
using PyPlot
using DataFrames
using CSV

include(pwd()*"/src/multivar alpha stable dist/set_up.jl")

y_test = Matrix(CSV.read("data/multivar alpha stable/y_test.csv"; allowmissing=:auto))

# results for DNN simple small

loss_training_dnn_simple = Matrix(CSV.read("data/multivar alpha stable/loss_vec_training_simple_DNN_small_1.csv"; allowmissing=:auto))
loss_val_dnn_simple = Matrix(CSV.read("data/multivar alpha stable/loss_vec_val_simple_DNN_small_1.csv"; allowmissing=:auto))
predictions_dnn_simple = Matrix(CSV.read("data/multivar alpha stable/predictions_simple_DNN_small_1.csv"; allowmissing=:auto))

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
PyPlot.figure()
PyPlot.plt[:hist2d](y_test[:,5], predictions_dnn_simple[:,5], bins=(100, 100))
PyPlot.figure()
PyPlot.plt[:hist2d](y_test[:,6], predictions_dnn_simple[:,6], bins=(100, 100))

# plot predictions
PyPlot.figure()
PyPlot.plot(y_test[:,1], predictions_dnn_simple[:,1], "*")
PyPlot.figure()
PyPlot.plot(y_test[:,2], predictions_dnn_simple[:,2], "*")
PyPlot.figure()
PyPlot.plot(y_test[:,3], predictions_dnn_simple[:,3], "*")
PyPlot.figure()
PyPlot.plot(y_test[:,4], predictions_dnn_simple[:,4], "*")
PyPlot.figure()
PyPlot.plot(y_test[:,5], predictions_dnn_simple[:,5], "*")
PyPlot.figure()
PyPlot.plot(y_test[:,6], predictions_dnn_simple[:,6], "*")


# check pred error
loss_test = sqrt.(sum((predictions_dnn_simple - y_test).^2, dims=2))

PyPlot.figure()
h = PyPlot.plt[:hist](loss_test,100)


# results for DNN timeserirs

loss_training_dnn_timseries_data = Matrix(CSV.read("data/multivar alpha stable/loss_vec_training_dnn_timseries_data_1.csv"; allowmissing=:auto))
loss_val_dnn_timseries_data = Matrix(CSV.read("data/multivar alpha stable/loss_vec_val_dnn_timseries_data_1.csv"; allowmissing=:auto))
predictions_dnn_timseries_data = Matrix(CSV.read("data/multivar alpha stable/predictions_dnn_timseries_data_1.csv"; allowmissing=:auto))

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
PyPlot.figure()
PyPlot.plt[:hist2d](y_test[:,5], predictions_dnn_timseries_data[:,5], bins=(100, 100))
PyPlot.figure()
PyPlot.plt[:hist2d](y_test[:,6], predictions_dnn_timseries_data[:,6], bins=(100, 100))

# plot predictions
PyPlot.figure()
PyPlot.plot(y_test[:,1], predictions_dnn_timseries_data[:,1], "*")
PyPlot.figure()
PyPlot.plot(y_test[:,2], predictions_dnn_timseries_data[:,2], "*")
PyPlot.figure()
PyPlot.plot(y_test[:,3], predictions_dnn_timseries_data[:,3], "*")
PyPlot.figure()
PyPlot.plot(y_test[:,4], predictions_dnn_timseries_data[:,4], "*")
PyPlot.figure()
PyPlot.plot(y_test[:,5], predictions_dnn_timseries_data[:,5], "*")
PyPlot.figure()
PyPlot.plot(y_test[:,6], predictions_dnn_timseries_data[:,6], "*")

# check pred error
loss_test = sqrt.(sum((predictions_dnn_timseries_data - y_test).^2, dims=2))

PyPlot.figure()
h = PyPlot.plt[:hist](loss_test,100)



# results for deepsets

loss_training_deepsets = Matrix(CSV.read("data/multivar alpha stable/loss_vec_training_deepsets_1.csv"; allowmissing=:auto))
loss_val_deepsets = Matrix(CSV.read("data/multivar alpha stable/loss_vec_val_deepsets_1.csv"; allowmissing=:auto))
predictions_deepsets = Matrix(CSV.read("data/multivar alpha stable/predictions_deepsets_1.csv"; allowmissing=:auto))

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
PyPlot.figure()
PyPlot.plt[:hist2d](y_test[:,5], predictions_deepsets[:,5], bins=(100, 100))
PyPlot.figure()
PyPlot.plt[:hist2d](y_test[:,6], predictions_deepsets[:,6], bins=(100, 100))

# plot predictions
PyPlot.figure()
PyPlot.plot(y_test[:,1], predictions_deepsets[:,1], "*")
PyPlot.figure()
PyPlot.plot(y_test[:,2], predictions_deepsets[:,2], "*")
PyPlot.figure()
PyPlot.plot(y_test[:,3], predictions_deepsets[:,3], "*")
PyPlot.figure()
PyPlot.plot(y_test[:,4], predictions_deepsets[:,4], "*")
PyPlot.figure()
PyPlot.plot(y_test[:,5], predictions_deepsets[:,5], "*")
PyPlot.figure()
PyPlot.plot(y_test[:,6], predictions_deepsets[:,6], "*")

# check pred error
loss_test = sqrt.(sum((predictions_deepsets - y_test).^2, dims=2))

PyPlot.figure()
h = PyPlot.plt[:hist](loss_test,100)
