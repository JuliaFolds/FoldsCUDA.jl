try
    using FoldsCUDATests
    true
catch
    false
end || begin
    push!(LOAD_PATH, @__DIR__)
    using FoldsCUDATests
end
