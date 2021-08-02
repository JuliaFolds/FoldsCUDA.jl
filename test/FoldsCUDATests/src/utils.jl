module Utils

using CUDA

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
