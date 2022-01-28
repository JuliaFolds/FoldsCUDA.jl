using CUDA
using Folds
using FoldsCUDA
using StaticArrays
using Transducers
using Transducers: whencompletebasecase

# TODO: pretty FLoops syntax
function countints_svector_functional(indices, ::Val{n}; ex = PreferParallel()) where {n}

    function init()
        # On initialization, create a `MVector` as a basecase-local
        # sub-histogram buffer `b`:
        zero(MVector{n,Int})
    end

    function inc!(b, i)
        @inbounds b[max(begin, min(i, end))] += 1
        b
    end

    function completebasecase(b)
        # After basecase computing is done, convert the buffer `b` to an
        # immutable value `SVector` to share the value across threads:
        SVector(b)
    end

    function combine(h, b)
        # Cross-thread reduction is simply point-wise addition:
        h .+ b
    end

    rf =
        inc! |>
        wheninit(init) |>
        whencompletebasecase(completebasecase) |>
        whencombine(combine)

    Folds.reduce(rf, indices, ex)
end
