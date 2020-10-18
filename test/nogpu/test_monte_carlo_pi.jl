module TestMonteCarloPi
using LiterateTest.AssertAsTest: @assert
include("../../examples/monte_carlo_pi.jl")
using Test
@test (πₐₚₚᵣₒₓ - π) / π < 1e-2
end  # module
