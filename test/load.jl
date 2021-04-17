try
    using FoldsCUDATests
    true
catch
    false
end || begin
    push!(LOAD_PATH, joinpath(@__DIR__, "FoldsCUDATests"))
    using FoldsCUDATests
end
