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

A fold executor implemented using CUDA.jl.

For more information about executor, see
[Transducers.jl's glossary section](https://juliafolds.github.io/Transducers.jl/dev/explanation/glossary/#glossary-executor)
and
[FLoops.jl's API section](https://juliafolds.github.io/FLoops.jl/dev/reference/api/#executor).

# Examples
```jldoctest
julia> using FoldsCUDA, Folds

julia> Folds.sum(1:10, CUDAEx())
55
```
"""
struct CUDAEx{K} <: Executor
    kwargs::K
end

popsimd(; simd = nothing, kwargs...) = kwargs

Transducers.transduce(xf, rf::RF, init, xs, exc::CUDAEx) where {RF} =
    transduce_cuda(xf, rf, init, xs; popsimd(; exc.kwargs...)...)

Transducers.executor_type(::CuArray) = CUDAEx
