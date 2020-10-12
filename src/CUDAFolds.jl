module CUDAFolds

export CUDAEx, foldx_cuda, transduce_cuda

import FLoops
using CUDA
using Core: Typeof
using Core.Compiler: return_type
using GPUArrays: @allowscalar
using InitialValues: InitialValue, asmonoid
using Transducers:
    Map, Reduced, Transducer, combine, complete, next, opcompose, reduced, start, unreduced

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
    replace(read(path, String), r"^```julia"m => "```jldoctest README")
end CUDAFolds

end
