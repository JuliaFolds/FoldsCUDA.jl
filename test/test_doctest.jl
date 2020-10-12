module TestDoctest

using CUDAFolds
using Documenter: doctest
using Test

@testset "doctest" begin
    doctest(CUDAFolds, manual = false)
end

end  # module
