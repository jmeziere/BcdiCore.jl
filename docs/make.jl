using Documenter, BcdiCore

makedocs(
    sitename="BcdiCore.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages = [
        "Main"=>"index.md",
        "Scaling"=>"scaling.md",
        "API"=>"api.md"
    ]
)

deploydocs(
    repo = "github.com/byu-cig/BcdiCore.jl.git",
)
