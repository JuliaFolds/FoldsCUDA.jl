module TestTypeChangingAccumulators

using Adapt
using CUDA
using Folds
using Folds.Testing: Testing, parse_tests
using FoldsCUDA
using Test
using Transducers

missing_if_odd(x) = isodd(x) ? missing : x

function test_sum()
    @test Folds.sum(missing_if_odd, 0:2:2^10, CUDAEx()) == sum(0:2:2^10)
    @test Folds.sum(missing_if_odd, 0:2:2^20, CUDAEx()) == sum(0:2:2^20)
    @test Folds.sum(missing_if_odd, 0:2^10, CUDAEx()) === missing
    @test Folds.sum(missing_if_odd, 0:2^20, CUDAEx()) === missing
end

partition_length_maximum(xs, ex = PreferParallel()) = Transducers.fold(
    max,
    xs |> ReducePartitionBy(identity, Map(_ -> 1)'(+), 0),
    ex;
    init = typemin(Int),
)

function test_partition_length_maximum()
    @testset "2^$e" for e in [5, 10, 15, 20, 25]
        xs = CUDA.rand(Bool, 2^e)
        @test partition_length_maximum(xs) == partition_length_maximum(collect(xs))
    end
end


rawdata = """
prod(y for x in 1:11 if isodd(x) for y in 1:x:x^2)
reduce(TeeRF(min, max), (2x for x in 1:10 if isodd(x)))
"""

testdata = parse_tests(rawdata, @__MODULE__)

upload(x) = adapt(CuArray, x)

test_with_sequential() = Testing.test_with_sequential(testdata, [CUDAEx()]; upload = upload)

end  # module
