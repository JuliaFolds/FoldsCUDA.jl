"""
    foldx_cuda(op[, xf], xs; init)
    transduce_cuda(op[, xf], init, xs)

Extended fold backed up by CUDA.
"""
(foldx_cuda, transduce_cuda)

foldx_cuda(op, xs; init = DefaultInit, kwargs...) =
    Transducers.fold(op, xs, CUDAEx(; kwargs...); init = init)

foldx_cuda(op, xf, xs; kwargs...) = foldx_cuda(op, xf(xs); kwargs...)

"""
    CUDAEx()

FLoops.jl executor implemented using CUDA.jl.
"""
struct CUDAEx{K} <: Executor
    kwargs::K
end

popsimd(; simd = nothing, kwargs...) = kwargs

Transducers.transduce(xf, rf::RF, init, xs, exc::CUDAEx) where {RF} =
    transduce_cuda(xf, rf, init, xs; popsimd(; exc.kwargs...)...)

Transducers.executor_type(::CuArray) = CUDAEx
