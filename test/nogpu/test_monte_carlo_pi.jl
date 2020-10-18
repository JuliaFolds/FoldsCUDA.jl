module TestMonteCarloPi
using LiterateTest.AssertAsTest: @assert
include("../../examples/monte_carlo_pi.jl")
@test (πₐₚₚᵣₒₓ - π) / π < 1e-2
end  # module
