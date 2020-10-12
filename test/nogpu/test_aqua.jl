module TestAqua

import Aqua
import CUDAFolds
using Test

Aqua.test_all(
    CUDAFolds;
    # Default `Aqua.test_all(CUDAFolds)` does not work due to ambiguities
    # in upstream packages:
    ambiguities = false,
    # Since CUDA.jl only supports Julia 1.5, there is no reason to
    # support `[extras]`:
    project_extras = false,
)

@testset "Method ambiguity" begin
    Aqua.test_ambiguities(CUDAFolds)
end

end  # module
