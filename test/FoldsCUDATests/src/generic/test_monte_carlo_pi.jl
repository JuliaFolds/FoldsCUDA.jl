module TestMonteCarloPi
using LiterateTest.AssertAsTest: @assert
test() = include(joinpath(@__DIR__, "../examples/monte_carlo_pi.jl"))
end  # module
