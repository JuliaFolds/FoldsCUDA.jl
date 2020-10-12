module TestAqua

import Aqua
import CUDAFolds
using Test

# Default `Aqua.test_all(CUDAFolds)` does not work due to ambiguities
# in upstream packages.
Aqua.test_all(CUDAFolds; ambiguities = false)

@testset "Method ambiguity" begin
    Aqua.test_ambiguities(CUDAFolds)
end

end  # module
