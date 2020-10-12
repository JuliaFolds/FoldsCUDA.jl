module TestCUDAFolds

using Test

const TEST_GPU = lowercase(get(ENV, "CUDAFOLDS_JL_TEST_GPU", "true")) == "true"

if TEST_GPU
    @testset "$file" for file in sort([
        file for file in readdir(@__DIR__) if match(r"^test_.*\.jl$", file) !== nothing
    ])
        include(file)
    end
end

@testset "$file" for file in sort([
    file
    for
    file in readdir(joinpath(@__DIR__, "nogpu")) if
    match(r"^test_.*\.jl$", file) !== nothing
])
    include(joinpath("nogpu", file))
end

end  # module
