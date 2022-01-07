# Implements a simple class for propagating uncertainties using
# the formalism in ISO GUM (JCGM 100 and JCGM 102).
using Printf
using LinearAlgebra
using DataFrames
using Formatting

"""
    checkcovariance!(cov::AbstractMatrix)

Checks whether the matrix meets the criteria for a covariance matrix.
(Square, non-negative diagonal elements, symmetric cov[r,c]==cov[c,r], and the
correlation coefficient between -1.0 and 1.0).  The tests are approximate
to handle rounding errors but when a symmetric pair of values are not precisely
equal they are set equal and when the correlation coefficient is slightly outside
[-1,1], it is restricted to within these bounds. Thus the input matrix can be
modified, in ways that are probably benign, by this "check" function.
"""
function checkcovariance!(cov::AbstractMatrix{T}, tol = convert(T,1.0e-6))::Bool where { T<: AbstractFloat }
    sz = size(cov)
    if length(sz) ≠ 2
        error("The covariance must be a matrix.")
    end
    if sz[1] ≠ sz[2]
        error("The covariance matrix must be square.")
    end
    for rc = 1:sz[1]
        if cov[rc, rc] < zero(T)
            error("The diagonal elements must all be non-negative. -> ", cov[rc, rc])
        end
    end
    for r = 1:sz[1]
        for c = 1:r-1
            if !isapprox(cov[r, c], cov[c, r], atol = tol * sqrt(cov[r, r] * cov[c, c]))
                error("The variances must be symmetric. -> ", cov[r, c], " ≠ ", cov[c, r])
            end
            cov[c, r] = cov[r, c] # Now force it precisely...
        end
    end
    for r = 1:sz[1]
        for c = 1:r-1
            cc = cov[r, c] / sqrt(cov[r, r] * cov[c, c])
            if abs(cc) > one(T) + convert(T,1.0e-6)
                error(
                    "The variances must have a correlation coefficient between -1.0 and 1.0 -> ",
                    cc,
                )
            end
            if abs(cc) > one(T)
                cc = max(-one(T), min(one(T), cc))
                cov[r, c] = (cov[c, r] = cc * sqrt(cov[r, r] * cov[c, c]))
            end
        end
    end
    true
end

"""
    UncertainValues

Represents a set of related variables with the covariance matrix that represents
the uncertainty relationships between the variables.
"""
struct UncertainValues
    labels::Dict{<:Label,Int}
    values::AbstractVector{Float64}
    covariance::AbstractMatrix{Float64}
    UncertainValues(
        labels::Dict{<:Label,Int},
        values::AbstractVector{Float64},
        covar::AbstractMatrix{Float64},
    ) = checkUVS!(labels, values, covar) ? new(labels, values, covar) : error("???")
end
"""
    uvs(labels::AbstractVector{<:Label}, values::AbstractVector{T}, covar::AbstractMatrix{T})  where { T <: Real }
    uvs(labels::AbstractVector{<:Label}, values::AbstractVector{T}, vars::AbstractVector{T}) where { T <: Real }
    uvs(values::Pair{<:Label,UncertainValue}...)
    uvs(value::Pair{<:Label,UncertainValue})
    uvs(values::Dict{<:Label,UncertainValue})

Various methods for constructing `UncertainValues` structures from varying types of inputs.
"""
function uvs(
    labels::AbstractVector{<:Label},
    values::AbstractVector{T},
    covar::AbstractMatrix{T},
) where { T<: Real }
    @assert length(labels) == length(values)
    @assert length(labels) == size(covar, 1)
    @assert length(labels) == size(covar, 2)
    return UncertainValues(
        Dict{Label,Int}(l => i for (i, l) in enumerate(labels)),
        Float64.(values),
        Float64.(covar)
    )
end

function uvs(
    labels::AbstractVector{<:Label},
    values::AbstractVector{T},
    vars::AbstractVector{T},
) where { T <: Real }
    @assert length(labels) == length(values)
    @assert length(labels) == length(vars)
    return uvs(labels, Float64.(values), diagm(Float64.(vars)))
end

uvs(values::Pair{<:Label,UncertainValue}...) = uvs(Dict(values))
uvs(value::Pair{<:Label,UncertainValue}) = uvs(Dict(value))
covariance(uvs::UncertainValues) = uvs.covariance

function uvs(values::Dict{<:Label,UncertainValue})
    labels, vals, vars = Vector{Label}(), Vector{Float64}(), Vector{Float64}()
    for (lbl, uv) in values
        push!(labels, lbl)
        push!(vals, value(uv))
        push!(vars, variance(uv))
    end
    return uvs(labels, vals, diagm(vars))
end

"""
    estimated(labels::Vector{<:Label}, samples::Matrix{Float64})

Estimate an UncertainValues based on a set of samples ('measurements').
Computes the mean values and the covariance matrix from the expectation values.
Each row `r` in samples represents a measured quantity as identified by `labels[r]`.
Each column represents a single set of measurements of all the labeled quantities.
"""
function estimated(labels::Vector{<:Label}, samples::Matrix{Float64})::UncertainValues
    @assert length(labels) == size(samples, 1) "label length must equal row count in estimated"
    nvars, nsamples = size(samples, 1), size(samples, 2)
    μ = mean.(samples[k, :] for k = 1:nvars)
    n = collect(samples[k, :] .- μ[k] for k = 1:nvars)
    Σ = Matrix{Float64}(undef, nvars, nvars)
    for j = 1:nvars, k = 1:j
        Σ[j, k] = (Σ[k, j] = dot(n[j], n[k]) / nsamples)
    end
    return uvs(labels, μ, Σ)
end

function checkUVS!(
    labels::Dict{<:Label,Int},
    values::AbstractVector{Float64},
    covar::AbstractMatrix{Float64},
)
    if length(labels) ≠ length(values)
        error("The number of labels does not match the number of values.")
    end
    if length(labels) ≠ size(covar)[1]
        error("The number of labels does not match the dimension of the covariance matrix.")
    end
    checkcovariance!(covar)
end

Base.haskey(uvs::UncertainValues, lbl::Label) = haskey(uvs.labels, lbl)

"""
    σ(lbl::Label, uvs::UncertainValues)

Returns the 1σ uncertainty associated with the specified label
"""
σ(uvs::UncertainValues, lbl::Label) = sqrt(variance(uvs, lbl))
σ(uvs::UncertainValues, lbl::Label, def) =
    haskey(uvs.labels, lbl) ? sqrt(variance(uvs, lbl)) : def

"""
    correlation(uvs::UncertainValues, a::Label, b::Label)

Returns the Pearson correlation coefficient between variables `a` and `b`.
"""
correlation(uvs::UncertainValues, a::Label, b::Label) =
    covariance(uvs, a, b) / (σ(uvs, a) * σ(uvs, b))

"""
    Base.filter(uvs::UncertainValues, labels::Vector{<:Label})::Matrix

Extract the covariance matrix associated with the variables specified in labels
into a Matrix.
"""
function Base.filter(uvs::UncertainValues, labels::Vector{<:Label})::Matrix
    idx = map(l -> uvs.labels[l], labels) # look it up once...
    m = zeros(length(idx), length(idx))
    for (r, rl) in enumerate(idx)
        for (c, cl) in enumerate(idx)
            m[c, r] = (m[r, c] = uvs.covariance[rl, cl])
        end
    end
    return m
end

Base.:*(aa::AbstractMatrix{T}, uvs::UncertainValues) where { T <: AbstractFloat } =
    UncertainValues(uvs.labels, aa * uvs.values, aa * uvs.covariance * transpose(aa))

Base.:*(aa::Diagonal{T}, uvs::UncertainValues) where { T <: AbstractFloat } =
    UncertainValues(uvs.labels, aa * uvs.values, aa * uvs.covariance * aa)


"""
    cat(uvss::AbstractArray{UncertainValues})::UncertainValues

Combines the disjoint UncertainValues in uvss into a single UncertainValues object.
"""
Base.cat(uvss::AbstractArray{UncertainValues})::UncertainValues = cat(uvss...)

"""
    cat(uvss::UncertainValues...)::UncertainValues

Combines the disjoint UncertainValues in uvss into a single UncertainValues object.
"""
function Base.cat(uvss::UncertainValues...)::UncertainValues
    function combinelabels(us::UncertainValues...) # Maintains order as union doesn't
        res = Label[]
        for u in us
            for lu in labels(u)
                if !(lu in res)
                    push!(res, lu)
                else
                    throw(
                        ErrorException(
                            "Unable to combine UncertainValues with duplicate labels.",
                        ),
                    )
                end
            end
        end
        return res
    end
    all = Dict{Label,Int}(lbl => i for (i, lbl) in enumerate(combinelabels(uvss...)))
    # @assert length(all) == sum(map(uvs -> length(uvs.labels), uvss)) "One or more labels were duplicated in cat(...)"
    len = length(all)
    values, covar = zeros(Float64, len), zeros(Float64, len, len)
    # Precompute the indexes for speed ( index in all, index in uvs)
    for uvs in uvss
        # Map  label indices into new label indices
        idx = collect((all[lbl], rc) for (lbl, rc) in uvs.labels)
        for (newR, oldR) in idx
            values[newR] = uvs.values[oldR]
            for (newC, oldC) in idx
                covar[newR, newC] = uvs.covariance[oldR, oldC]
            end
        end
    end
    return UncertainValues(all, values, covar)
end

Base.show(io::IO, uvs::UncertainValues) = print(
    io,
    "UVS[" *
    join(
        (
            @sprintf("%s = %-0.3g ± %-0.3g", lbl, value(uvs, lbl), σ(uvs, lbl)) for
            lbl in labels(uvs)
        ),
        ", ",
    ) *
    "]",
)

function Base.show(io::IO, ::MIME"text/plain", uvs::UncertainValues)
    trim(str, len) = str[1:min(len, length(str))] * " "^max(0, len - min(len, length(str)))
    lbls = sortedlabels(uvs)
    print(io, "Variable       Value              ")
    foreach(l -> print(io, trim(repr(l), 12)), lbls)
    for (r, rl) in enumerate(lbls)
        println(io)
        print(io, trim("$rl", 10) * @sprintf(" | %-8.3g |", value(uvs, rl)))
        print(io, r == length(uvs.labels)[1] / 2 ? "  ±  |" : "     |")
        foreach(cl -> print(io, @sprintf("   %-8.3g ", covariance(uvs, rl, cl))), lbls)
        print(io, " |")
    end
end

Base.show(io::IO, m::MIME"text/html", uvs::UncertainValues) =
# We piggyback off DataFrames here because generating a good looking table isn't easy...
    show(io, m, asa(DataFrame, uvs), summary = false, eltypes = false)

function Base.show(io::IO, ::MIME"text/markdown", uvs::UncertainValues)
    fmt(x) = @sprintf("%0.2e", x)
    cv(r, c) = r == c ? "($(fmt(σ(uvs, r))))²" : "$(fmt(covariance(uvs,r,c)))"
    esc(ss) = replace(ss, "|" => "\\|")
    ext(ss, l) = ss * repeat(" ", l - length(ss))
    lbls, rows = labels(uvs), []
    push!(rows, ["Labels", "Values", "", repr.(lbls)...])
    for lbl in lbls
        push!(
            rows,
            [repr(lbl), fmt(value(uvs, lbl)), " ", (cv(lbl, col) for col in lbls)...],
        )
    end
    cl = maximum(length(rows[j][i]) for j in eachindex(rows) for i in eachindex(rows[j]))
    insert!(
        rows,
        2,
        [
            ":$(repeat("-",cl-2))-",
            ":$(repeat("-",cl-2)):",
            ":$(repeat("-",cl-2)):",
            (":$(repeat("-",cl-2)):" for l in lbls)...,
        ],
    )
    rows[(length(rows)-3)÷2+3][3] = ext("±", cl)
    ss = join(
        (
            "| " * join((ext(rows[r][c], cl) for c in eachindex(rows[r])), " | ") * " |" for r in eachindex(rows)
        ),
        "\n",
    )
    print(io, ss)
end

"""
    sortedlabels(uvs::UncertainValues)

A alphabetically sorted list of the labels. Warning this can be slow.  Use keys(...) if you
want just a unordered set of labels.
"""
sortedlabels(uvs::UncertainValues) =
    sort([keys(uvs.labels)...], lt = (l, m) -> isless(repr(l), repr(m)))

Base.keys(uvs::UncertainValues) = keys(uvs.labels)

function Base.getindex(uvs::UncertainValues, lbl::Label)::UncertainValue
    idx = uvs.labels[lbl]
    return UncertainValue(uvs.values[idx], sqrt(uvs.covariance[idx, idx]))
end

function Base.isnan(uvs::UncertainValues, lbl::Label)::Bool
    idx = uvs.labels[lbl]
    return isnan(uvs.values[idx]) || isnan(uvs.covariance[idx,idx])
end

function Base.get(
    uvs::UncertainValues,
    lbl::Label,
    def::Union{Missing,UncertainValue},
)::Union{Missing,UncertainValue}
    idx = get(uvs.labels, lbl, -1)
    return idx ≠ -1 ? UncertainValue(uvs.values[idx], sqrt(uvs.covariance[idx, idx])) : def
end

Base.length(uvs::UncertainValues) = length(uvs.labels)

eachlabel(uvs::UncertainValues) = keys(uvs.labels)

"""
    labels(uvs::UncertainValues)::Vector{<:Label}

Returns a Vector of Label in the order in which they appear in `uvs.values` and
`uvs.covariance`.
"""
function labels(uvs::UncertainValues)::Vector{<:Label}
    res = Array{Label}(undef, length(uvs.labels))
    for (k, v) in uvs.labels
        res[v] = k
    end
    return res
end

"""
    value(uvs::UncertainValues, lbl::Label)
    value(uvs::UncertainValues, lbl::Label, defValue)

The value associate with the Label. The first implementation throws an exception if `lbl` is not present
while the second implementation returns `defValue`
"""
value(uvs::UncertainValues, lbl::Label) = uvs.values[uvs.labels[lbl]]
value(uvs::UncertainValues, lbl::Label, defValue) =
    haskey(uvs.labels, lbl) ? uvs.values[uvs.labels[lbl]] : defValue

"""
    values(uvs::UncertainValues)::Vector{Float64}

A Vector containing the Float64 value for each row in uvs.  In the same order as
labels(uvs).
"""
Base.values(uvs::UncertainValues) = uvs.values

"""
   covariance(uvs::UncertainValues, lbl1::Label, lbl2::Label)

The covariance between the two variables.
"""
covariance(uvs::UncertainValues, lbl1::Label, lbl2::Label) =
    uvs.covariance[uvs.labels[lbl1], uvs.labels[lbl2]]

covariance(uvs::UncertainValues, lbl1::Label, lbl2::Label, default) =
    haskey(uvs.labels, lbl1) && haskey(uvs.labels, lbl2) ? #
    uvs.covariance[uvs.labels[lbl1], uvs.labels[lbl2]] :
    default

"""
   variance(uvs::UncertainValues, lbl::Label)
   variance(uvs::UncertainValues, lbl::Label, default)

The variance associated with the specified Label.
"""
variance(uvs::UncertainValues, lbl::Label) =
    max(0.0, uvs.covariance[uvs.labels[lbl], uvs.labels[lbl]])
variance(uvs::UncertainValues, lbl::Label, default) =
    haskey(uvs.labels, lbl) ? uvs.covariance[uvs.labels[lbl], uvs.labels[lbl]] : default


function asa( #
    ::Type{DataFrame},
    uvss::UncertainValues,
    withCovars = true,
)::DataFrame
    lbls = labels(uvss)
    df = DataFrame(Variable = map(lbl -> "$lbl", lbls), Values = map(lbl -> value(uvss, lbl), lbls))
    if withCovars
        for (i, cl) in enumerate(lbls)
            insertcols!(
                df,
                2 + i,
                Symbol("$cl") => map(rl -> covariance(uvss, rl, cl), lbls),
            )
        end
    else
        insertcols!(df, 3, :σ => map(lbl -> σ(uvss, lbl), lbls))
    end
    return df
end

"""
    uncertainty(lbl::Label, uvs::UncertainValues, k::Float64=1.0)

The uncertainty associated with specified label (k σ where default k=1)
"""
uncertainty(uvs::UncertainValues, lbl::Label, k::Float64 = 1.0) = k * σ(uvs, lbl)

Base.indexin(uvs::UncertainValues, lbl::Label) = uvs.labels[lbl]

"""
    labeledvalues(uvs::UncertainValues)

Converts the values of an UncertainValues object into a LabeledValues object.
"""
labeledvalues(uvs::UncertainValues) = LabeledValues(labels(uvs), uvs.values)

labelsByType(ty::Type{<:Label}, uvs::UncertainValues) =
    filter(lbl -> typeof(lbl) == ty, labels(uvs))

function labelsByType(types::AbstractVector{DataType}, uvs::UncertainValues)
    lbls = labels(uvs)
    mapreduce(ty -> labelsByType(ty, lbls), append!, types)
end


"""
    extract(uvss::UncertainValues, labels::AbstractVector{<:Label})::UncertainValues
    extract(uvss::UncertainValues, labeltype::Type{T})::UncertainValues where {T<:Label}
    extract(uvss::UncertainValues, labeltypes::AbstractVector{DataType})::UncertainValues

Construct an `UncertainValues` datum containing only the specified labels/labeltype(s).
"""

function extract(uvss::UncertainValues, labels::AbstractVector{<:Label})::UncertainValues
    idx = map(l -> indexin(uvss, l), labels) # look it up once...
    vals = Float64[uvss.values[i] for i in idx]
    cov = zeros(length(idx), length(idx))
    for (r, rl) in enumerate(idx)
        for (c, cl) in enumerate(idx)
            cov[c, r] = (cov[r, c] = uvss.covariance[rl, cl])
        end
    end
    return uvs(labels, vals, cov)
end

function extract(
	uvss::UncertainValues,
    labeltype::Type{T}
)::UncertainValues where {T<:Label}
    return extract(uvss, labelsByType(labeltype, uvss))
end

function extract(
	uvss::UncertainValues,
    labeltypes::AbstractVector{DataType},
)::UncertainValues
    return extract(uvss, labelsByType(labeltypes, uvss))
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