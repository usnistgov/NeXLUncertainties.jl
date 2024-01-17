module NeXLUncertaintiesLaTeXStringsExt

using LaTeXStrings, NeXLUncertainties, Printf

"""
    LaTeXStrings.latexstring(uv::UncertainValue; fmt=nothing, mode=:normal[|:siunitx])::LaTeXString

Converts an `UncertainValue` to a `LaTeXString` in a reasonable manner.
`mode=:siunitx" requires `\\usepackage{siunitx}` which defines `\\num{}`.
`fmt` is a C-style format string like "%0.2f" or nothing for an "intelligent" default.
"""
function LaTeXStrings.latexstring(
    uv::UncertainValue;
    fmt = nothing,
    mode = :normal,
)::LaTeXString
    pre, post = (mode == :siunitx ? ("\\num{", "}") : ("", ""))
    if isnothing(fmt)
        return latexstring(pre * NeXLUncertainties._showstr(uv, raw"\pm") * post)
    else
        fmtrfunc(x) = Printf.format(Printf.Format(fmt), x)
        return latexstring(
            pre * fmtrfunc(value(uv)) * " \\pm " * fmtrfunc(uncertainty(uv)) * post,
        )
    end
end

"""

    LaTeXStrings.latexstring(uvs::UncertainValues; fmt="%0.3f", mode=:normal[|:siunutx])::LaTeXString

Formats a `UncertainValues` as a LaTeXString (`mode=:siunitx` uses `\\num{}` from the siunitx package.)
Also requires the amsmath package for `vmatrix`
"""
function LaTeXStrings.latexstring(uvs::UncertainValues; fmt="%0.3f", mode=:normal)::LaTeXString
    fmtrfunc = generate_formatter( fmt )
    lbls = labels(uvs)
    pre, post = ( mode == :siunitx ? ( raw"\num{", raw"}" ) : ( "", "" ) )
    ls = join([ 
            raw"\begin{vmatrix}", 
            join([ repr(lbl) for lbl in lbls ], " \\\\\n"), 
            raw"\end{vmatrix}" 
        ],"\n") 
    vals = join([ 
            raw"\begin{vmatrix}", 
            join([ pre*fmtrfunc(value(uvs, lbl))*post for lbl in lbls ], " \\\\\n"), 
            raw"\end{vmatrix}" 
        ],"\n") 
    covs = join([
        raw"\begin{vmatrix}", 
        join( [
            join( [ pre*fmtrfunc(covariance(uvs, lbl1, lbl2))*post for lbl1 in lbls ], " & ") 
                for lbl2 in lbls ], " \\\\\n"),
        raw"\end{vmatrix}" ],"\n")
    return latexstring("$ls\n=$vals\n\\pm\n$covs")
end

end # module