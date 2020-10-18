using CUDA
using FLoops
using FoldsCUDA
using Random123

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
