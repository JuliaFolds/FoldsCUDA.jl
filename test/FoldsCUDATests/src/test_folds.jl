module TestReduce

using CUDA
using Folds
using Test

function test_sum_pairs()
    xs = CUDA.rand(Int32, 100)
    @test Folds.sum(last, pairs(xs); init = Int32(0)) == sum(xs)
    VERSION >= v"1.6-" || return
    @test Folds.sum(last, pairs(xs)) == sum(xs)
end

end  # module
