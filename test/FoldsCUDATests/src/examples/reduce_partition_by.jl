macro inbounds(ex)
    esc(ex)
end

include(joinpath(@__DIR__, "../../../../examples/reduce_partition_by.jl"))

using Test
@testset begin
    @test sum(c_xs) == length(xs)
    @test sum(m_xs .* c_xs) ≈ sum(xs)
end
