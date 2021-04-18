# # `findminmax`

using Transducers
using CUDA
using FoldsCUDA
using FLoops

function findminmax(xs, ex = xs isa CuArray ? CUDAEx() : ThreadedEx())
    xtypemax = typemax(eltype(xs))
    xtypemin = typemin(eltype(xs))
    @floop ex for (i, x) in pairs(xs)
        @reduce() do (imin = -1; i), (xmin = xtypemax; x)
            if xmin > x
                xmin = x
                imin = i
            end
        end
        @reduce() do (imax = -1; i), (xmax = xtypemin; x)
            if xmax < x
                xmax = x
                imax = i
            end
        end
    end
    return (; imin, xmin, imax, xmax)
end

function findminmax_base(xs)
    xmin, imin = findmin(xs)
    xmax, imax = findmax(xs)
    return (; imin, xmin, imax, xmax)
end
nothing  # hide
#-

xs = [700, 900, 500, 200, 700, 700, 900, 300, 600, 400, 900, 600, 900, 800, 600]
if has_cuda_gpu()
    xs = CuArray(xs)
end

result = findminmax(xs)
#-

@assert result == findminmax_base(xs)
