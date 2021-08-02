module TestReducePartitionBy
using ...Utils

_test() = include(joinpath(@__DIR__, "../examples/reduce_partition_by.jl"))

function test()
    try
        _test()
    finally
        Utils.unsafe_free_all!(@__MODULE__)
    end
end

end  # module
