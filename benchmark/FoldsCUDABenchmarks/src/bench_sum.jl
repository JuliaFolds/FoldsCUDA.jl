module BenchSum

using BenchmarkTools
using CUDA
using Folds
using FoldsCUDA

const CACHE = Ref{Any}(nothing)

fact(::Val{0}) = 1
fact(::Val{N}) where {N} = N * fact(Val(N - 1))

""" Some function with controllable complexity. """
function exp_approx(x, ::Val{N}) where {N}
    coefficients = ntuple(p -> 1.0 / fact(Val(p - 1)), Val(N))
    return evalpoly(x, coefficients)
end

function setup(; test = false, n = test ? 10 : 2^30, exp_approx_n = 10)
    CACHE[] = CUDA.randn(n)

    suite = BenchmarkGroup()

    s = suite["identity"] = BenchmarkGroup()
    s["base"] = @benchmarkable sum(CACHE[])
    s["folds"] = @benchmarkable Folds.sum(CACHE[])
    s["coalesced"] = @benchmarkable Folds.sum(CACHE[], CoalescedCUDAEx())

    ea = let ea, n = Val(exp_approx_n)
        ea(x) = exp_approx(x, n)
    end
    s = suite["exp_approx"] = BenchmarkGroup()
    s["base"] = @benchmarkable sum($ea, CACHE[])
    s["folds"] = @benchmarkable Folds.sum($ea, CACHE[])
    s["coalesced"] = @benchmarkable Folds.sum($ea, CACHE[], CoalescedCUDAEx())

    s = suite["sin"] = BenchmarkGroup()
    s["base"] = @benchmarkable sum(sin, CACHE[])
    s["folds"] = @benchmarkable Folds.sum(sin, CACHE[])
    s["coalesced"] = @benchmarkable Folds.sum(sin, CACHE[], CoalescedCUDAEx())

    return suite
end

function clear()
    xs = CACHE[]
    CACHE[] = nothing
    if xs isa CuArray
        CUDA.unsafe_free!(xs)
    end
end

end  # module BenchSum
