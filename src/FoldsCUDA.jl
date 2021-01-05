module FoldsCUDA

export CUDAEx, foldx_cuda, transduce_cuda

using CUDA
using Core: Typeof
using Core.Compiler: return_type
using GPUArrays: @allowscalar
using InitialValues: InitialValue, asmonoid
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
    unreduced

# TODO: Don't import internals from Transducers:
using Transducers:
    DefaultInit,
    DefaultInitOf,
    EmptyResultError,
    IdentityTransducer,
    _reducingfunction,
    extract_transducer

include("kernels.jl")
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
