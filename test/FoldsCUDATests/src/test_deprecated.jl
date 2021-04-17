module TestDeprecated

using CUDA
using FoldsCUDA
using Test
using Transducers

function test_foldx_cuda()
    result = Ref{Any}(nothing)
    xs = CUDA.ones(3)
    @test_deprecated result[] = foldx_cuda(TeeRF(min, max), xs)
    if result[] !== nothing
        @test result[] == (1, 1)
    end
end

function test_transduce_cuda()
    result = Ref{Any}(nothing)
    xs = CUDA.ones(3)
    @test_deprecated result[] =
        transduce_cuda(Map(identity), TeeRF(min, max), (0.0f0, 2.0f0), xs)
    if result[] !== nothing
        @test result[] == (0, 2)
    end
end

end  # module
