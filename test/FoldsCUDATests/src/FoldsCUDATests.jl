module FoldsCUDATests

using GPUArrays
using Test
using TestFunctionRunner

include("utils.jl")

module TestGeneric
using ..Utils: include_tests
include_tests(@__MODULE__, joinpath(@__DIR__, "generic"))
end

module TestGPU
using ..Utils: include_tests, should_test_gpu
include_tests(@__MODULE__)
should_test_module() = should_test_gpu()
end

""" A list of tests to be run from UnionArrays.jl """
const UNIONARRAYS_TESTS = [TestGPU.TestTypeChangingAccumulators]

function runtests_unionarrays(; kwargs...)
    TestFunctionRunner.run(UNIONARRAYS_TESTS; kwargs...)
end

function __init__()
    # TODO: add pre/post test hook to TestFunctionRunner?
    GPUArrays.allowscalar(false)
end

end # module
