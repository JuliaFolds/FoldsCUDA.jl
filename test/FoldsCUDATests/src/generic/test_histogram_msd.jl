module TestHistogramMSD
using LiterateTest.AssertAsTest: @assert
test() = include(joinpath(@__DIR__, "../examples/histogram_msd.jl"))
end  # module
