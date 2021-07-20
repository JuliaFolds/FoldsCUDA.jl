module TestBenchmarks

using FoldsCUDABenchmarks
using Test

function test()
    @test try
        suite = FoldsCUDABenchmarks.setup_smoke()
        run(suite)
        true
    finally
        FoldsCUDABenchmarks.clear()
    end
end

end  # module
