module TestAqua

import Aqua
import FoldsCUDA
using Test

Aqua.test_all(
    FoldsCUDA;
    # Default `Aqua.test_all(FoldsCUDA)` does not work due to ambiguities
    # in upstream packages:
    ambiguities = false,
    # Since CUDA.jl only supports Julia 1.5, there is no reason to
    # support `[extras]`:
    project_extras = false,
)

@testset "Method ambiguity" begin
    Aqua.test_ambiguities(FoldsCUDA)
end

end  # module
