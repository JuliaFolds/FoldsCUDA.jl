try
    using FoldsCUDATests
    true
catch
    false
end || begin
    let path = joinpath(@__DIR__, "FoldsCUDATests")
        path in LOAD_PATH || push!(LOAD_PATH, path)
    end
    let path = joinpath(@__DIR__, "../benchmark/FoldsCUDABenchmarks")
        path in LOAD_PATH || push!(LOAD_PATH, path)
    end
    using FoldsCUDATests
end
