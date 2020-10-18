# # Estimating π using Monte-Carlo method

# Let's compute an approximation of π using the [Monte Carlo
# method](https://en.wikipedia.org/wiki/Monte_Carlo_method).  The idea
# is to draw points from the uniform distribution on a unit square and
# count the ratio of points that are inside the quadrant of the unit
# circle.  Since the probably ``p`` for a point to be inside the
# quadrant is the area of the quadrant (``= π / 4``) divided by the
# area of the unit square (``= 1``), we can estimate ``π`` by
# estimating the probability ``p = π / 4``:

xs = rand(10_000)
ys = rand(10_000)
p = count(xs.^2 .+ ys.^2 .< 1) / length(xs)
4 * p

# > ![Illustration of monte-Carlo method for computing pi](https://upload.wikimedia.org/wikipedia/commons/8/84/Pi_30K.gif)
# >
# > --- [File:Pi 30K.gif - Wikipedia](https://en.wikipedia.org/wiki/File:Pi_30K.gif)
# > by
# > [nicoguaro](https://commons.wikimedia.org/wiki/User:Nicoguaro)
# > is licensed under
# > [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/deed.en).

# We try to do this computation on a GPU using FoldsCUDA.jl:

using CUDA
using FLoops
using FoldsCUDA

# As of writing, `CUDA.CURAND` does not provide the API usable inside
# the loop body (i.e., the device API).  However, we can use
# pure-Julia pseudo number generator quite easily.  In particular, we
# use [Counter-based random number generator
# (CBRNG)](https://en.wikipedia.org/wiki/Counter-based_random_number_generator_(CBRNG))
# provided by [Random123.jl](https://github.com/sunoru/Random123.jl)
# ([documentation](https://sunoru.github.io/RandomNumbers.jl/stable/man/random123/)).

using Random123

# In this example, we use
# [`Random123.Philox2x`](https://sunoru.github.io/RandomNumbers.jl/stable/lib/random123/#Random123.Philox2x).
# This RNG gives us two `UInt64`s for each counter which wraps around
# at `typemax(UInt64)`:

rng_a = Philox2x(0)
rng_b = Philox2x(0)
set_counter!(rng_b, typemax(UInt64))
rand(rng_b, UInt64, 2)
@assert rng_a == rng_b

# Let's create a helper function that divides
# `UInt64(0):typemax(UInt64)` into `n` equally spaced points:

function counters(n)
    stride = typemax(UInt64) ÷ n
    return UInt64(0):stride:typemax(UInt64)-stride
end
nothing  # hide
#-

# This let us use independent RNG for each `ctr`-th iteration:

function monte_carlo_pi(n, m = 10_000, ex = has_cuda_gpu() ? CUDAEx() : ThreadedEx())
    @floop ex for ctr in counters(n)
        rng = set_counter!(Philox2x(0), ctr)
        nhits = 0
        for _ in 1:m
            nhits += rand(rng)^2 + rand(rng)^2 < 1
        end
        @reduce(tot = 0 + nhits)
    end
    return 4 * tot / (n * m)
end
nothing  # hide
#-

πₐₚₚᵣₒₓ = monte_carlo_pi(2^12)
