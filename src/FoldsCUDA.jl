module FoldsCUDA

export CUDAEx, CoalescedCUDAEx, foldx_cuda, transduce_cuda

using CUDA
using CUDA: @allowscalar
using Core: Typeof
using Core.Compiler: return_type
using InitialValues: InitialValue, asmonoid
using UnionArrays: UnionArrays, UnionVector
using Transducers:
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
    extract_transducer,
    foldl_nocomplete

include("kernels.jl")
include("shfl.jl")
include("api.jl")

# Use README as the docstring of the module:
@doc let path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    doc = read(path, String)
    doc = replace(doc, r"^```julia"m => "```jldoctest README")
    doc = replace(doc, "(https://juliafolds.github.io/FoldsCUDA.jl/dev/examples/)" => "(@ref examples-toc)")
    doc
end FoldsCUDA

end
