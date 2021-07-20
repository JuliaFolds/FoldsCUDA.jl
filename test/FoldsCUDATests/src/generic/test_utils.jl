module TestUtils

using FoldsCUDA: ithtype
using Test

function test_ithtype()
    for n in 2:5
        T = Union{(Val{i} for i in 1:n)...}
        @test Union{(ithtype(T, Val(i)) for i in 1:n)...} == T
    end
end

end  # module
