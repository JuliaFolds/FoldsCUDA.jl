macro manual_union_split(body::Expr, conditions...)
    if !(body.head === :-> && length(body.args) == 2 && body.args[1] == :(()))
        error(
            "`@manual_union_split` is intended to be used with a `do` blockt",
            " with no argumen",
        )
    end
    body = body.args[2]
    ex = foldr(conditions; init = body) do c, ex
        quote
            if $c
                $body
            else
                $ex
            end
        end
    end
    esc(ex)
end

valueof(::Val{x}) where {x} = x

function ithtype(::Type{T}, i::Val) where {T}
    S = foldrunion(T, Val(1)) do S, j
        if j === i
            S
        else
            if j isa Val
                Val(valueof(j) + 1)
            else
                j
            end
        end
    end
    if S isa Type
        return S
    else
        return Union{}
    end
end
