module TestFoldsCUDA

using Test

if VERSION ≥ v"1.6-"
    try
        using CUDA
    catch
        @info "Failed to import CUDA. Trying again with `@stdlib`..."
        push!(LOAD_PATH, "@stdlib")
    end
end
using CUDA

const TEST_GPU =
    lowercase(get(ENV, "JULIA_PKGEVAL", "false")) != "true" &&
    lowercase(get(ENV, "CUDAFOLDS_JL_TEST_GPU", "true")) == "true"

@testset "$file" for file in sort([
    file for file in readdir(@__DIR__) if match(r"^test_.*\.jl$", file) !== nothing
])
    TEST_GPU || continue  # branch inside `for` loop for printing skipped tests
    include(file)
end

@testset "nogpu/$file" for file in sort([
    file
    for
    file in readdir(joinpath(@__DIR__, "nogpu")) if
    match(r"^test_.*\.jl$", file) !== nothing
])
    include(joinpath("nogpu", file))
end

end  # module
