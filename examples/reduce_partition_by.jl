# # Partition reduce on GPU

# Using
# [`ReducePartitionBy`](https://juliafolds.github.io/Transducers.jl/dev/reference/manual/#Transducers.ReducePartitionBy)
# per-partition (group) can be computed on GPU in a streaming (single-pass)
# fashion.

using CUDA
using CUDA: @allowscalar

n = 2^24
if has_cuda_gpu()
    xs = CUDA.randn(n)
else
    xs = randn(n)
end
nothing # hide

# `ReducePartitionBy` expects partition to be continuous; i.e., sorted by the
# key. We will use `floor` as the key. So, plain `sort!` works in this example.

sort!(xs)
nothing # hide

# In GPU, it is convenient to know the output location before computation. So,
# let us build unique index for each partition using `cumsum!`:

function buildindices(f, xs)
    isedge(x, y) = !isequal(f(x), f(y))
    bounds = similar(xs, Bool)
    @views map!(isedge, bounds[2:end], xs[1:end-1], xs[2:end])
    @allowscalar bounds[1] = true
    partitionindices = similar(xs, Int32)
    return cumsum!(partitionindices, bounds)
end

partitionindices_xs = buildindices(floor, xs)
nothing # hide

# ### Counting the size of each partition

import FoldsCUDA  # register the executor
using FLoops
using Transducers

function countparts(partitionindices; ex = nothing)
    nparts = @allowscalar partitionindices[end]
    ys = similar(partitionindices, nparts)

    ## The intra-partition reducing function that reduces each partition to
    ## a 2-tuple of index and count:
    rf_partition = Map(p -> (p, 1))'(ProductRF(right, +))

    index_and_count =
        partitionindices |>
        ReducePartitionBy(
            identity,  # partition by partitionindices
            rf_partition,
            (-1, 0),
        )

    @floop ex for (p, c) in index_and_count
        @inbounds ys[p] = c
    end

    return ys
end

c_xs = countparts(partitionindices_xs)
#-

# ### Computing the average of each partition

function meanparts(xs, partitionindices; ex = nothing)
    nparts = @allowscalar partitionindices[end]
    ys = similar(xs, float(eltype(xs)), nparts)

    ## The intra-partition reducing function that reduces each partition to
    ## a 3-tuple of index, count and sum:
    rf_partition = Map(((i, p),) -> (p, 1, (@inbounds xs[i])))'(ProductRF(right, +, +))

    index_count_and_sum =
        pairs(partitionindices) |>
        ReducePartitionBy(
            ((_, p),) -> p,  # partition by partitionindices
            rf_partition,
            (-1, 0, zero(eltype(ys))),
        )

    @floop ex for (p, c, s) in index_count_and_sum
        @inbounds ys[p] = s / c
    end

    return ys
end

m_xs = meanparts(xs, partitionindices_xs)
#-
