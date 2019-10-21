# Implements a simple class for propagating uncertainties using
# the formalism in ISO GUM (JCGM 100 and JCGM 102).
using SparseArrays
using Printf
using LinearAlgebra
using DataFrames

"""
    checkcovariance!(cov::AbstractMatrix)

Checks whether the matrix meets the criteria for a covariance matrix.
(Square, non-negative diagonal elements, symmetric cov[r,c]==cov[c,r], and the
correlation coefficient between -1.0 and 1.0).  The tests are approximate
to handle rounding errors but when a symmetric pair of values are not precisely
equal they are set equal and when the correlation coefficient is slightly outside
[-1,1], it is restricted to within these bounds. Thus the input matrix can be
modified by this "check" function.
"""
function checkcovariance!(cov::AbstractMatrix{Float64}, tol = 1.0e-12)::Bool
    sz = size(cov)
    if length(sz) ≠ 2
        error("The covariance must be a matrix.")
    end
    if sz[1] ≠ sz[2]
        error("The covariance matrix must be square.")
    end
    for rc = 1:sz[1]
        if cov[rc, rc] < 0.0
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
            if abs(cc) > 1.0 + 1.0e-12
                error("The variances must have a correlation coefficient between -1.0 and 1.0 -> ", cc)
            end
            if abs(cc) > 1.0
                cc = max(-1.0, min(1.0, cc))
                cov[r, c] = (cov[c, r] = cc * sqrt(cov[r, r] * cov[c, c]))
            end
        end
    end
    true
end

"""
    checkcovariance!(cov::SparseMatrixCSC{Float64,Int})

Checks whether the sparse matrix meets the criteria for a covariance matrix.
(Square, non-negative diagonal elements, symmetric cov[r,c]==cov[c,r], and the
correlation coefficient between -1.0 and 1.0).  The tests are approximate
to handle rounding errors but when a symmetric pair of values are not precisely
equal they are set equal and when the correlation coefficient is slightly outside
[-1,1], it is restricted to within these bounds. Thus the input matrix can be
modified by this "check" function.
"""
function checkcovariance!(cov::SparseMatrixCSC{Float64,Int}, tol = 1.0e-12)::Bool
    # The generic AbstractMatrix implementation can be really slow on large sparce matrices.
    sz = size(cov)
    if length(sz) ≠ 2
        error("The covariance must be a matrix.")
    end
    if sz[1] ≠ sz[2]
        error("The covariance matrix must be square.")
    end
    for rc = 1:sz[1]
        if cov[rc, rc] < 0.0
            error("The diagonal elements must all be non-negative. (S) -> ", cov[rc, rc])
        end
    end
    for ci in findall(!iszero, cov)
        r, c = ci[1], ci[2]
        if !isapprox(cov[ci], cov[c, r], atol = abs(tol) * sqrt(cov[r, r] * cov[c, c]))
            error("The variances must be symmetric. (S) -> ", cov[ci], " ≠ ", cov[c, r])
        end
        cov[c, r] = cov[ci] # Now force it precisely...
    end
    for ci in findall(!iszero, cov)
        r, c = ci[1], ci[2]
        cc = cov[ci] / sqrt(cov[r, r] * cov[c, c])
        if abs(cc) > 1.0 + 1.0e-12
            error("The variances must have a correlation coefficient between -1.0 and 1.0 (S) -> ", cc)
        end
        if abs(cc) > 1.0
            cc = max(-1.0, min(1.0, cc))
            cov[ci] = (cov[c, r] = cc * sqrt(cov[r, r] * cov[c, c]))
        end
    end
    true
end

abstract type Label end

struct BasicLabel{T} <: Label
    value::T
end

label(item::T) where {T} = BasicLabel(item)

Base.show(io::IO, sl::BasicLabel) = print(io, "Label[$(repr(sl.value))]")
Base.isequal(sl1::BasicLabel{T}, sl2::BasicLabel{T}) where {T} = isequal(sl1.value, sl2.value)
Base.isequal(sl1::Label, sl2::Label) = false

"""
    UncertainValues

Represents a set of related variables with the covariance matrix that represents
the uncertainty relationships between the variables.
"""
struct UncertainValues
    labels::Dict{<:Label,Int}
    values::AbstractVector{Float64}
    covariance::AbstractMatrix{Float64}
    UncertainValues(labels::Dict{<:Label,Int}, values::AbstractVector{Float64}, covar::AbstractMatrix{Float64}) =
        checkUVS!(labels, values, covar) ? new(labels, values, covar) : error("???")
end

uvs(labels::AbstractVector{<:Label}, values::AbstractVector{Float64}, covar::AbstractMatrix{Float64}) =
    UncertainValues(Dict{Label,Int}([(l, i) for (i, l) in enumerate(labels)]), values, covar)

function checkUVS!(labels::Dict{<:Label,Int}, values::AbstractVector{Float64}, covar::AbstractMatrix{Float64})
    if length(labels) ≠ length(values)
        error("The number of labels does not match the number of values.")
    end
    if length(labels) ≠ size(covar)[1]
        error("The number of labels does not match the dimension of the covariance matrix.")
    end
    checkcovariance!(covar)
end
"""
    σ(lbl::Label, uvs::UncertainValues)

Returns the 1σ uncertainty associated with the specified label
"""
σ(lbl::Label, uvs::UncertainValues) = sqrt(variance(lbl, uvs))


"""
    extract(uvs::UncertainValues, labels::Vector{<:Label})::Matrix

Extract the covariance matrix associated with the variables specified in labels
into a Matrix.
"""
function extract(uvs::UncertainValues, labels::Vector{<:Label})::Matrix
    m = zeros(length(labels), length(labels))
    for (r, rl) in enumerate(labels)
        for (c, cl) in enumerate(labels)
            m[r, c] = covariance(rl, cl, uvs)
        end
    end
    return m
end

Base.:*(aa::AbstractMatrix{Float64}, uvs::UncertainValues) =
    UncertainValues(uvs.labels, aa * uvs.values, aa * uvs.covariance * transpose(aa))

Base.:*(aa::Diagonal{Float64}, uvs::UncertainValues) =
    UncertainValues(uvs.labels, aa * uvs.values, aa * uvs.covariance * aa)

"""
    cat(uvss::AbstractArray{UncertainValues})::UncertainValues

Combines the disjoint UncertainValues in uvss into a single UncertainValues object.
"""
function Base.cat(uvss::AbstractArray{UncertainValues})::UncertainValues
    all = Dict{Label,Int}()
    next = 1
    for uvs in uvss
        for lbl in labels(uvs)
            if !haskey(all, lbl)
                all[lbl] = next
                next += 1
            end
        end
    end
    len = length(all)
    values, covar = zeros(Float64, len), zeros(Float64, len, len)
    for uvs in uvss
        for rlbl in labels(uvs)
            ridx = all[rlbl]
            values[ridx] = value(rlbl, uvs)
            for clbl in labels(uvs)
                cidx = all[clbl]
                covar[ridx, cidx] = covariance(rlbl, clbl, uvs)
            end
        end
    end
    checkcovariance!(covar)
    UncertainValues(all, values, covar)
end

function Base.show(io::IO, uvs::UncertainValues)
    trim(str, len) = str[1:min(len, length(str))] * " "^max(0, len - min(len, length(str)))

    lbls = labels(uvs)
    print(io, "Variabl    ")
    print(io, "    Value    ")
    print(io, "          ")
    for l in lbls
        print(io, trim(repr(l), 10))
        print(io, "  ")
    end
    println(io)

    for (r, rl) in enumerate(lbls)
        print(io, trim(repr(rl), 10))
        print(io, @sprintf(" | %-8.3g |", value(rl, uvs)))
        print(io, r == length(uvs.labels)[1] / 2 ? "  ±  |" : "     |")
        for cl in lbls
            print(io, @sprintf("   %-8.3g ", covariance(rl, cl, uvs)))
        end
        println(io, " |")
    end
end

"""
    labels(uvs::UncertainValues)

A alphabetically sorted list of the labels
"""
labels(uvs::UncertainValues) = sort([keys(uvs.labels)...], lt = (l, m) -> isless(repr(l), repr(m)))

function Base.getindex(uvs::UncertainValues, lbl::Label)::UncertainValue
    idx = uvs.labels[lbl]
    return UncertainValue(uvs.values[idx], sqrt(uvs.covariance[idx, idx]))
end

function Base.get(uvs::UncertainValues, lbl::Label, def::Union{Missing,UncertainValue})::UncertainValue
    idx = get(uvs.labels, lbl, -1)
    return idx ≠ -1 ? UncertainValue(uvs.values[idx], sqrt(uvs.covariance[idx, idx])) : def
end

Base.length(uvs::UncertainValues) = length(uvs.labels)
Base.size(uvs::UncertainValues) = size(uvs.values)

"""
    value(lbl::Label, uvs::UncertainValues)

The value associate with the Label.
"""
value(lbl::Label, uvs::UncertainValues) = uvs.values[uvs.labels[lbl]]

"""
    values(uvs::UncertainValues)

A Dict containing <:Label => UncertainValue for each row in uvs.
"""
Base.values(uvs::UncertainValues) = Dict((lbl, uvs[lbl]) for lbl in keys(uvs.labels))

"""
   covariance(lbl1::Label, lbl2::Label, uvs::UncertainValues)

The covariance between the two variables.
"""
covariance(lbl1::Label, lbl2::Label, uvs::UncertainValues) = uvs.covariance[uvs.labels[lbl1], uvs.labels[lbl2]]

"""
   variance(lbl::Label, uvs::UncertainValues)

The variance associated with the specified Label.
"""
variance(lbl::Label, uvs::UncertainValues) = uvs.covariance[uvs.labels[lbl], uvs.labels[lbl]]

function tabulate(uvss::AbstractVector{UncertainValues}, withUnc = false)::DataFrame
    val(uv) = ismissing(uv) ? missing : uv.value
    sig(uv) = ismissing(uv) ? missing : uv.sigma
    alllabels = sort(mapreduce(labels, union, uvss), lt = (l, m) -> isless(repr(l), repr(m)))
    df = DataFrame()
    for lbl in alllabels
        df[!, Symbol(repr(lbl))] = [val(get(uvs, lbl, missing)) for uvs in uvss]
        if withUnc
            df[!, Symbol("U($(repr(lbl)))")] = [sig(get(uvs, lbl, missing)) for uvs in uvss]
        end
    end
    return df
end


"""
    uncertainty(lbl::Label, uvs::UncertainValues, k::Float64=1.0)

The uncertainty associated with specified label (k σ where default k=1)
"""
uncertainty(lbl::Label, uvs::UncertainValues, k::Float64 = 1.0) =
    k * sqrt(uvs.covariance[uvs.labels[lbl], uvs.labels[lbl]])

struct Jacobian
    entries::AbstractMatrix{Float64}
    inputs::Dict{<:Label,Int}
    outputs::Dict{<:Label,Int}
    Jacobian(input::AbstractArray{<:Label}, output::AbstractArray{<:Label}, entries::AbstractMatrix{Float64}) =
        (size(entries)[2] == length(input)) && (size(entries)[1] == length(output)) ?
        new(entries, buildDict(input), buildDict(output)) :
        error("The output and input lengths must match the row and column dimensions of the matrix.")
end

inputLabels(jac::Jacobian) = keys(jac.inputs)

outputLabels(jac::Jacobian) = keys(jac.outputs)

"""
    propagate(jac::Jacobian, uvs::UncertainValues)::Matrix

Propagate the covariance matrix in uvs using the specified Jacobian creating a new covariance matrix.
C' = J⋅C⋅transpose(J)
"""
function propagate(jac::Jacobian, uvs::UncertainValues)::Matrix
    function extract(jac::Jacobian, uvs::UncertainValues)::Matrix
        res = zeros(size(jac.entries)[1], length(uvs.labels))
        for (l, c) in uvs.labels
            res[:, c] = jac.entries[:, jac.inputs[l]]
        end
        res
    end
    j = extract(jac, uvs)
    j * uvs.covariance * transpose(j)
end
