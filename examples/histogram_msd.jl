# # Histogram of the most significant digit (MSD)

using LiterateTest                                                     #src
using Test                                                             #src

# ## An allocation-free function to compute MSD

function msd(x::Real)
    x = abs(x)
    d = x
    while true
        x < 1 && return floor(Int, d)
        d = x
        x ÷= 10
    end
end
nothing  # hide
#-

@test begin
    msd(34513)
end == 3
#-

@test begin
    msd(-51334)
end == 5
#-

@assert 2.76e19 > typemax(Int64)  #src
@test begin
    msd(2.76e19)
end == 2
#-

# ## MSD of random numbers

using CUDA
using FLoops
using FoldsCUDA
using Setfield

function histogram_msd(xs, ex = xs isa CuArray ? CUDAEx() : ThreadedEx())
    zs = ntuple(_ -> 0, 9)  # a tuple of 9 zeros
    @floop ex for x in xs
        d = msd(x)
        1 <= d <= 9 || continue  # skip it if `msd` returns 0
        h2 = @set zs[d] = 1      # set `d`-th position of the tuple to 1
        @reduce(h1 = zs .+ h2)   # point-wise addition merges the histogram
    end
    return h1
end
nothing  # hide

# Generate some random numbers

xs = let randn = has_cuda_gpu() ? CUDA.randn : randn
    exp.(10.0 .* (randn(10^8) .+ 6))
end
@assert all(isfinite, xs)
collect(view(xs, 1:10))  # preview

# Pass an array of (real) numbers to `histogram_msd` to compute the
# histogram of MSD:

hist1 = histogram_msd(xs)

# Frequency in percentage:

aspercentage(h) = pairs(round.((collect(h) ./ length(xs) .* 100); digits = 1))
aspercentage(hist1)
#-

# ## MSD of lazily computed numbers

# FoldsCUDA supports running reduction over non-`CuArray` containers
# such as `UnitRange` and iterator transformation wrapping it.
# However, you need to explicitly specify to use CUDA by, e.g.,
# passing `CUDAEx` to `@floop`:

hist2 = histogram_msd((x^2 for x in 1:10^8), CUDAEx())

# Frequency in percentage:
aspercentage(hist2)
#-

hist3 = histogram_msd((exp(x) for x in range(1, 35, length=10^8)), CUDAEx())

# Frequency in percentage:
aspercentage(hist3)
