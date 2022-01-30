module FoldsCUDA

export CUDAEx, CoalescedCUDAEx, foldx_cuda, transduce_cuda

using CUDA
using CUDA: @allowscalar
using Core: Typeof
using InitialValues: InitialValue, asmonoid
using UnionArrays: UnionArrays, UnionVector
using Transducers:
    @return_if_reduced,
    Executor,
    Map,
    Reduced,
    Transducer,
    Transducers,
    combine,
    complete,
    next,
    opcompose,
    reduced,
    start,
    transduce,
    unreduced

# TODO: Don't import internals from Transducers:
using Transducers:
    AbstractReduction,
    DefaultInit,
    DefaultInitOf,
    EmptyResultError,
    IdentityTransducer,
    Reduction,
    _reducingfunction,
    completebasecase,
    extract_transducer,
    foldl_basecase

include("utils.jl")
include("kernels.jl")
include("unionvalues.jl")
include("shfl.jl")
include("api.jl")
include("introspection.jl")

# Use README as the docstring of the module:
@doc let path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    doc = read(path, String)
    doc = replace(doc, r"^```julia"m => "```jldoctest README")
    doc = replace(doc, "(https://juliafolds.github.io/FoldsCUDA.jl/dev/examples/)" => "(@ref examples-toc)")
    doc
end FoldsCUDA

end
