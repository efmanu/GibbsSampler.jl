using Documenter, GibbsSampler

makedocs(
    modules = [GibbsSampler],
    format = Documenter.HTML(),
    sitename = "GibbsSampler.jl",
    doctest = true,
    pages = [
        "Home" => "index.md",
        "Sampling Algorithm" => "structs.md",
        "Examples" => "examples.md",
        "Comparison" => "compare.md"
    ]
)

deploydocs(
    repo = "github.com/efmanu/GibbsSampler.jl.git",
)