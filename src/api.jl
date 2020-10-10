"""
    foldx_cuda(op[, xf], xs; init)
    transduce_cuda(op[, xf], init, xs)

Extended fold backed up by CUDA.
"""
(foldx_cuda, transduce_cuda)

foldx_cuda(op, xs; init = DefaultInit, kwargs...) =
    unreduced(transduce_cuda(op, init, xs; kwargs...))

foldx_cuda(op, xf, xs; init = DefaultInit, kwargs...) =
    unreduced(transduce_cuda(xf, op, init, xs; kwargs...))

"""
    CUDAEx()

FLoops.jl executor implemented using CUDA.jl.
"""
struct CUDAEx{K} <: FLoops.Executor
    kwargs::K
end

popsimd(; simd = nothing, kwargs...) = kwargs

FLoops._fold(rf::RF, init, xs, exc::CUDAEx) where {RF} =
    foldx_cuda(rf, IdentityTransducer(), xs; popsimd(; exc.kwargs...)..., init = init)
