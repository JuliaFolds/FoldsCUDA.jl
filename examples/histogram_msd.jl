# # Histogram of the most significant digit

using CUDA
using FLoops
using FoldsCUDA
using Setfield

function msd(x::Integer)
    x = abs(x)
    d = x
    while true
        x == 0 && return d
        d = x
        x ÷= 10
    end
end

function msd(x::Real)
    x = abs(x)
    while x > typemax(Int64)
        x ÷= 1000000000000000000
    end
    return msd(floor(Int64, x))
end

function histogram_msd(xs, ex = xs isa CuArray ? CUDAEx() : ThreadedEx())
    zs = ntuple(_ -> 0, 9)
    @floop ex for x in xs
        d = msd(x)
        1 <= d <= 9 || continue
        h2 = @set zs[d] = 1
        @reduce(h1 = zs .+ h2)
    end
    return h1
end

xs = let randn = has_cuda_gpu() ? CUDA.randn : randn
    exp.(10.0 .* (randn(10^8) .+ 6))
end
@assert all(isfinite, xs)

hist = histogram_msd(xs)
#-

pairs(round.((collect(hist) ./ length(xs) .* 100); digits = 1))
#-
