using Documenter, NeXLUncertainties

rm("build", force=true, recursive=true)

include("../weave/buildweave.jl")

makedocs(
    modules = [NeXLUncertainties],
    sitename = "NeXLUncertainties.jl",
    pages = [ "Home" => "index.md", "Getting Started" => "gettingstarted.md", "Resistor Network Example"=>"resistors.md", "Methods" => "methods.md" ]
)

rm("src/gettingstarted.md", force=true)
rm("src/resistors.md", force=true)
rm("src/resistor.png", force=true)

# deploydocs(repo = "github.com/NeXLUncertainties.jl.git")
