# # Computing π using Monte-Carlo method

using CUDA
using FLoops
using FoldsCUDA

# [Random123.jl](https://github.com/sunoru/Random123.jl)
# [Counter-based random number generator (CBRNG)](https://en.wikipedia.org/wiki/Counter-based_random_number_generator_(CBRNG))
# [documentation](https://sunoru.github.io/RandomNumbers.jl/stable/man/random123/)

using Random123

# In this example, we use
# [`Random123.Philox2x`](https://sunoru.github.io/RandomNumbers.jl/stable/lib/random123/#Random123.Philox2x)
# whose period is `typemax(UInt64)` (by default).

function counters(n)
    stride = typemax(UInt64) ÷ n
    return UInt64(0):stride:typemax(UInt64)-stride
end

function monte_carlo_pi(n, m = 10_000, ex = has_cuda_gpu() ? CUDAEx() : ThreadedEx())
    @floop ex for ctr in counters(n)
        rng = set_counter!(Philox2x(0), ctr)
        nhits = 0
        for _ in 1:m
            x = 2(rand(rng) - 0.5)
            y = 2(rand(rng) - 0.5)
            nhits += x^2 + y^2 < 1
        end
        @reduce(tot = 0 + nhits)
    end
    return 4 * tot / (n * m)
end

πₐₚₚᵣₒₓ = monte_carlo_pi(2^12)
