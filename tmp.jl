using Transducers
include("examples/findminmax.jl")

results = 1:10000 |> KeepSomething() do seed
    CUDA.seed!(seed)
    n = 513
    xs = CUDA.rand(n)
    println()
    println()
    println()
    @show seed
    a = findminmax(xs)
    b = findminmax(Array(xs))
    if a != b
        return (a,b,xs)
    end
end |> Take(1) |> collect
