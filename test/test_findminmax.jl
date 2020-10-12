module TestFindminmax

using Test
using Random
include("../examples/findminmax.jl")

@testset for seed in 1:100
    CUDA.seed!(seed)
    rng = MersenneTwister(seed)
    # ns = [1:10; [255, 256, 257, 511, 512, 513]; rand(11:1000, 100); ]
    ns = [1:1100; [10^4, 10^5, 10^6]]
    @testset for n in ns
        xs = CUDA.rand(n)
        xs_cpu = Array(xs)
        @test findminmax(xs) == findminmax(xs_cpu) == findminmax_base(xs_cpu)
    end
end

@testset "large" begin
    seed = 123456789
    CUDA.seed!(seed)
    rng = MersenneTwister(seed)
    @testset for n in [10^7, 10^8, 10^9]
        GC.gc(true)
        @info "Ran `GC.gc(true)` (n = $n)"
        CUDA.memory_status()
        xs = CUDA.rand(n)
        xs_cpu = Array(xs)
        @test findminmax(xs) == findminmax(xs_cpu) == findminmax_base(xs_cpu)
    end
end

end  # module
