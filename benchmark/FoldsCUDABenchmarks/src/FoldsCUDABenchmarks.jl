module FoldsCUDABenchmarks

using BenchmarkTools: Benchmark, BenchmarkGroup

include("bench_sum.jl")

function setup(; kwargs...)
    suite = BenchmarkGroup()
    suite["sum"] = BenchSum.setup(; kwargs...)
    return suite
end

function set_smoke_params!(bench)
    bench.params.seconds = 0.001
    bench.params.evals = 1
    bench.params.samples = 1
    bench.params.gctrial = false
    bench.params.gcsample = false
    return bench
end

foreach_benchmark(f!, bench::Benchmark) = f!(bench)
function foreach_benchmark(f!, group::BenchmarkGroup)
    for x in values(group)
        foreach_benchmark(f!, x)
    end
end

function setup_smoke()
    suite = setup(test = true)
    foreach_benchmark(set_smoke_params!, suite)
    return suite
end

function clear()
    BenchSum.clear()
end

end  # module FoldsCUDABenchmarks
