module TestFolds

using CUDA
using Folds
using FoldsCUDA
using Referenceables
using Test

function test_sum_pairs()
    xs = CUDA.rand(Int32, 100)
    @test Folds.sum(last, pairs(xs); init = Int32(0)) == sum(xs)
    VERSION >= v"1.7-" || return
    # Non-type-stable reduction:
    @test Folds.sum(last, pairs(xs)) == sum(xs)
end

function inc!(xs, ex = CUDAEx())
    Folds.foreach(referenceable(xs), ex) do ref
        ptr = pointer(ref.x, LinearIndices(ref.x)[ref.i...])
        CUDA.atomic_add!(ptr, one(ref[]))
    end
    return xs
end

""" Test that each side effect is executed exactly once.  """
function test_side_effects()
    @testset for ex in [CUDAEx(), CoalescedCUDAEx()]
        test_side_effects(ex)
    end
end

function test_side_effects(ex)
    # Using an array size large enough s.t. it uses two blocks but not fully
    # occupy all threads.
    xs = inc!(CUDA.ones(1500), ex)

    @test all(==(2), xs)
end

end  # module
