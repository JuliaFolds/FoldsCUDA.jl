module TestTypeChangingAccumulators

using Folds
using FoldsCUDA
using Test

missing_if_odd(x) = isodd(x) ? missing : x

@testset begin
    @test Folds.sum(missing_if_odd, 0:2:2^10, CUDAEx()) == sum(0:2:2^10)
    @test Folds.sum(missing_if_odd, 0:2:2^20, CUDAEx()) == sum(0:2:2^20)
end

end  # module
