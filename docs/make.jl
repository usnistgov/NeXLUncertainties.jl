using Documenter, NeXLUncertainties

rm("build", force=true, recursive=true)

include("../weave/buildweave.jl")

makedocs(
    modules = [NeXLUncertainties],
    sitename = "NeXLUncertainties.jl",
    pages = [ "Home" => "index.md", "Getting Started" => "gettingstarted.md", "Methods" => "methods.md" ]
)

rm("src/gettingstarted.md", force=true)

# deploydocs(repo = "github.com/NeXLUncertainties.jl.git")
