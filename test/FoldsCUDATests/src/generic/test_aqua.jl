module TestAqua

import Aqua
import FoldsCUDA

test_all() = Aqua.test_all(
    FoldsCUDA;
    # Default `Aqua.test_all(FoldsCUDA)` does not work due to ambiguities
    # in upstream packages:
    ambiguities = false,
    # support `[extras]`:
    project_extras = false,
)

test_ambiguities() = Aqua.test_ambiguities(FoldsCUDA)

end  # module
