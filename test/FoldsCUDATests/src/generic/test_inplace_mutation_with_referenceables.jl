module TestInplaceMutationWithReferenceables
using LiterateTest.AssertAsTest: @assert
test() = include(joinpath(@__DIR__, "../examples/inplace_mutation_with_referenceables.jl"))
end  # module
