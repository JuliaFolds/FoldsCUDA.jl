module TestHistograms

include("../../../examples/histograms.jl")
using Test

function test_one_to_ten()
    indices = CuVector(1:10)
    h = countints_svector_functional(indices, Val(10))
    @test collect(h) == fill(1, 10)
end

function test_rand()
    indices = (floor(Int, x * 10) + 1 for x in CUDA.rand(2^30))
    h = countints_svector_functional(indices, Val(10))
    h = collect(h)
    @test h ./ h[1] ≈ fill(1, 10) atol = 0.01
end

end  # module
