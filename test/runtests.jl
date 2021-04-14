if VERSION ≥ v"1.6-"
    try
        pkgid = Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA")
        Base.require(pkgid)
    catch
        @info "Failed to import CUDA. Trying again with `@stdlib`..."
        push!(LOAD_PATH, "@stdlib")
    end
end
include("load.jl")
FoldsCUDATests.runtests()
