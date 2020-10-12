module TestDoctest

using FoldsCUDA
using Documenter: doctest
using Test

@testset "doctest" begin
    doctest(FoldsCUDA, manual = false)
end

end  # module
