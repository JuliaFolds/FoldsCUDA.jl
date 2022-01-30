module Utils

using CUDA

function include_tests(m, dir = @__DIR__)
    for file in readdir(dir)
        if match(r"^test_.*\.jl$", file) !== nothing
            Base.include(m, joinpath(dir, file))
        end
    end
end

should_test_gpu() =
    lowercase(get(ENV, "JULIA_PKGEVAL", "false")) != "true" &&
    lowercase(get(ENV, "CUDAFOLDS_JL_TEST_GPU", "true")) == "true"

"""
    unsafe_free_all!(namespace::Module)

Call `CUDA.unsafe_free!` on all `CuArray`s in `namespace`.
"""
function unsafe_free_all!(namespace::Module)
    for n in names(namespace; all = true)
        x = try
            getfield(namespace, n)
        catch err
            err isa UndefVarError || rethrow()
            continue
        end
        if x isa CuArray
            @debug "Free: `$namespace.$n`"
            CUDA.unsafe_free!(x)
        end
    end
end

end  # module
