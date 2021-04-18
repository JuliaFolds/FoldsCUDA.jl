using CUDA
using CUDA: @allowscalar

n = 2^20
if has_cuda_gpu()
    xs = CUDA.randn(n)
else
    xs = randn(n)
end

function buildindices(f, xs)
    isedge(x, y) = !isequal(f(x), f(y))
    bounds = similar(xs, Bool, n)
    @views map!(isedge, bounds[2:end], xs[1:end-1], xs[2:end])
    @allowscalar bounds[1] = true
    return cumsum(bounds)
end

sort!(xs)
partitionindices_xs = buildindices(floor, xs)


using Transducers
using Folds
using FLoops

function countparts(partitionindices)
    nparts = @allowscalar partitionindices[end]
    ys = similar(partitionindices, nparts)

    index_and_count =
        partitionindices |> ReducePartitionBy(
            identity,
            Map(p -> (p, 1))'(ProductRF(right, +)),
            (-1, 0),
        )
    @floop nothing for (p, c) in index_and_count
        @inbounds ys[p] = c
    end
    return ys
end

c_xs = countparts(partitionindices_xs)
#-

function meanparts(xs, partitionindices)
    nparts = @allowscalar partitionindices[end]
    ys = similar(xs, float(eltype(xs)), nparts)

    index_count_and_sum =
        pairs(partitionindices) |> ReducePartitionBy(
            ((_, p),) -> p,
            Map(((i, p),) -> (p, 1, (@inbounds xs[i])))'(ProductRF(right, +, +)),
            (-1, 0, zero(eltype(ys))),
        )
    @floop nothing for (p, c, s) in index_count_and_sum
        @inbounds ys[p] = s / c
    end
    return ys
end

m_xs = meanparts(xs, partitionindices_xs)
#-
