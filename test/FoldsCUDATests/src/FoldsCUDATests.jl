module FoldsCUDATests

import CUDA
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

function before_test_module()
    GPUArrays.allowscalar(false)

    if lowercase(get(ENV, "CI", "false")) == "true"
        CUDA.versioninfo()
        println()
    end
end

end # module
