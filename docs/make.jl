using Documenter, BcdiCore

makedocs(
    sitename="BcdiCore.jl",
    pages = [
        "Main"=>"index.md",
        "Scaling"=>"scaling.md",
        "API"=>"api.md"
    ]
)

deploydocs(
    repo = "github.com/byu-cig/BcdiCore.jl.git",
)
