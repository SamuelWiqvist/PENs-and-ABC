println("start gpu test")

using Pkg

println("build Knet")
Pkg.build("Knet")

using Knet 

println("test gpu")
println(Knet.gpuCount())
println(Knet.gpu())
Knet.gpu(0)
println(Knet.gpu())

Pkg.test("Knet")

println("set weights")
scaling = Float32(0.1)

w = Any[ scaling*randn(Float32,5,8),   zeros(Float32,5,1),
         xavier(Float32,1,5),    zeros(Float32,1,1) ]

println(typeof(w))
println(w)

println("map to Knet")

w = map(KnetArray,w)

println(typeof(w))
println(w)
println(w[1])


println("data")
 
x = rand(Float32, 5,10)

println(typeof(x))
println(x)

println("map to Knet")

x = convert(Knet.KnetArray,x)

println(typeof(x))
println(x)

println("more tests")


a = KnetArray(rand(Float32,3,3))

println(a)
println(a[[1,3],:])

b = relu.(a)

println(b)

println("end gpu test")
