using Documenter, NeXLUncertainties

function addNISTHeaders(htmlfile::String)
    # read HTML
    html = transcode(String,read(htmlfile))
    # Find </head>
    i = findfirst(r"</[Hh][Ee][Aa][Dd]>", html)
    # Already added???
    j = findfirst("nist-header-footer", html)
    if isnothing(j) && (!isnothing(i))
        # Insert nist-pages links right before </head>
        res = html[1:i.start-1]*
            "<link rel=\"stylesheet\" href=\"https://pages.nist.gov/nist-header-footer/css/nist-combined.css\">\n"*
            "<script src=\"https://pages.nist.gov/nist-header-footer/js/jquery-1.9.0.min.js\" type=\"text/javascript\" defer=\"defer\"></script>\n"*
            "<script src=\"https://pages.nist.gov/nist-header-footer/js/nist-header-footer.js\" type=\"text/javascript\" defer=\"defer\"></script>\n"*
            html[i.start:end]
        write(htmlfile, res)
        println("Inserting NIST header/footer into $htmlfile")
    end
    return htmlfile
end

rm("build", force=true, recursive=true)

include("../weave/buildweave.jl")

makedocs(
    modules = [NeXLUncertainties],
    sitename = "NeXLUncertainties.jl",
    pages = [ "Home" => "index.md",
              "Getting Started" => "gettingstarted.md",
              "Composing Multi-Step Models" => "composing.md",
              "Resistor Network Example"=>"resistors.md",
              "Methods" => "methods.md" ]
)

names = ( "gettingstarted", "composing", "resistors" )
addNISTHeaders(joinpath(@__DIR__, "build","index.html"))
addNISTHeaders.(map(name->joinpath(@__DIR__, "build", splitext(name)[1], "index.html"), names))

rm("src/gettingstarted.md", force=true)
rm("src/resistors.md", force=true)
rm("src/resistor.png", force=true)

# deploydocs(repo = "github.com/NeXLUncertainties.jl.git")
