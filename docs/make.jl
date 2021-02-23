using Documenter
using FoldsCUDA
using FLoops
using Literate
using LiterateTest
using LoadAllPackages

LoadAllPackages.loadall(joinpath((@__DIR__), "Project.toml"))

PAGES = [
    "index.md",
    "Examples" => [
        "`findminmax`" => "examples/findminmax.md",
        "Histogram of MSD" => "examples/histogram_msd.md",
        "Monte-Carlo π" => "examples/monte_carlo_pi.md",
        "In-place mutation with Referenceables.jl" =>
            "examples/inplace_mutation_with_referenceables.md",
    ],
]

let example_dir = joinpath(dirname(@__DIR__), "examples")
    examples = Pair{String,String}[]

    for subpages in PAGES
        subpages isa Pair || continue
        subpages[1] == "Examples" || continue
        for (_, mdpath) in subpages[2]::Vector
            stem, _ = splitext(basename(mdpath))
            jlpath = joinpath(example_dir, "$stem.jl")
            if !isfile(jlpath)
                @info "`$jlpath` does not exist. Skipping..."
                continue
            end
            push!(examples, jlpath => joinpath(@__DIR__, "src", dirname(mdpath)))
        end
    end

    @info "Compiling example files" examples
    for (jlpath, dest) in examples
        Literate.markdown(jlpath, dest; config = LiterateTest.config(), documenter = true)
    end
end


write(
    joinpath(@__DIR__, "src", "examples.md"),
    """
    # [Examples](@id examples-toc)
    ```@contents
    Pages = $(repr(last.(only(p for p in PAGES if p isa Pair && p[1] == "Examples")[2])))
    Depth = 3
    ```
    """,
)

makedocs(
    # See:
    # https://juliadocs.github.io/Documenter.jl/stable/lib/public/#Documenter.makedocs
    sitename = "FoldsCUDA",
    format = Documenter.HTML(),
    modules = [FoldsCUDA],
    pages = PAGES,
    doctest = false,
    strict = true,
)

deploydocs(
    # See:
    # https://juliadocs.github.io/Documenter.jl/stable/lib/public/#Documenter.deploydocs
    repo = "github.com/JuliaFolds/FoldsCUDA.jl",
    push_preview = true,
)
