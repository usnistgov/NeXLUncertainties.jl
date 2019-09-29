# Implements a simple class for propagating uncertainties using
# the formalism in ISO GUM (JCGM 100 and JCGM 102).
using SparseArrays
using Printf
using LinearAlgebra

"""
    UncertainValue

Represents a floating point numerical value and an associated uncertainty (1σ).
"""
struct UncertainValue <: AbstractFloat
    value::Float64
    sigma::Float64
    UncertainValue(val::Real, sigma::Real) = convert(Float64,sigma)>=0.0 ? new(convert(Float64, val), convert(Float64, sigma)) : error("σ must be non-negative.")
end

Base.convert(uvt::Type{UncertainValue}, n::Real) =
    typeof(n)==UncertainValue ? n : UncertainValue(convert(Float64, n),0.0)

Base.zero(::Type{UncertainValue}) = UncertainValue(0.0,0.0)
Base.one(::Type{UncertainValue}) = UncertainValue(1.0,0.0)

variance(uv::UncertainValue) = uv.sigma^2

σ(uv::UncertainValue) = uv.sigma

"""
    uncertainty(uv::UncertainValue, k::Real=1.0)

Returns the k-σ uncertainty (defaults to k=1.0)
"""
uncertainty(uv::UncertainValue, k::Real=1) = k*uv.sigma

"""
    fractional(uv::UncertainValue)

Computes the fractional uncertainty.
"""
fractional(uv::UncertainValue)::Float64 = uv.sigma/uv.value

"""
    value(uv::UncertainValue)

Returns the value portion. (uv.value)
"""
value(uv::UncertainValue) = uv.value

bound(n::Real, min::Real, max::Real) =
    n < min ? min : (n > max ? max : n)

multiply(a::Real, B::UncertainValue) =
    UncertainValue(convert(Float64,a)*B.value, abs(convert(Float64,a)*B.sigma))

multiply(A::UncertainValue, b::Real) =
    UncertainValue(convert(Float64,b)*A.value, abs(convert(Float64,b)*A.sigma))

Base.:*(a::Real, B::UncertainValue) = multiply(a,B)
Base.:*(A::UncertainValue, b::Real) = multiply(A,b)

Base.:-(A::UncertainValue) =
    UncertainValue(-A.value,A.sigma)
Base.:+(A::UncertainValue) =
    UncertainValue(A.value,A.sigma)

Base.abs(A::UncertainValue) =
    A.value>=0.0 ? A : UncertainValue(abs(A.value),A.sigma)

divide(A::UncertainValue, b::Real) =
    UncertainValue(A.value/convert(Float64,b), abs(A.sigma/convert(Float64,b)))

divide(a::Real, B::UncertainValue) =
    UncertainValue(convert(Float64,a)/B.value, abs((convert(Float64,a)*B.sigma)/(B.value*B.value)))

Base.:/(a::Real, B::UncertainValue) = divide(a,B)
Base.:/(A::UncertainValue, b::Real) = divide(A,b)

Base.inv(B::UncertainValue) =
    UncertainValue(1.0/B.value, abs(B.sigma/(B.value^2)))

power(A::UncertainValue, b::Real) =
    UncertainValue(A.value^b, abs(A.value^(b-one(b)) * b * A.sigma))

Base.:^(A::UncertainValue, b::Real) =
    power(A,b)

Base.log(A::UncertainValue) =
    UncertainValue(log(A.value), abs(A.sigma/A.value))

Base.exp(A::UncertainValue) =
    UncertainValue(exp(A.value), exp(A.value)*A.sigma)

Base.sin(A::UncertainValue) =
    UncertainValue(sin(A.value), abs(cos(A.value)*A.sigma))

Base.cos(A::UncertainValue) =
    UncertainValue(cos(A.value), abs(sin(A.value)*A.sigma))

Base.tan(A::UncertainValue) =
    UncertainValue(tan(A.value), sec(A.value)^2 * A.sigma)

Base.sqrt(A::UncertainValue) =
    UncertainValue(sqrt(A.value), abs(A.value^(-0.5) * 0.5 * A.sigma))

Base.isequal(A::UncertainValue, B::UncertainValue)::Bool =
    isequal(A.value, B.value) && isequal(A.sigma, B.sigma)

Base.isapprox(A::UncertainValue, B::UncertainValue; atol::Float64=0.0, rtol::Float64 = atol==0.0 ? 0.0 : √eps(Float64))::Bool =
    isapprox(A.value, B.value, rtol=rtol, atol=atol) && isapprox(A.sigma, B.sigma, rtol=rtol, atol=atol)

"""
    equivalent(A::UncertainValue, B::UncertainValue, k=1.0)

A and B are equivalent if |A-B| <= k σ(A-B)
"""
equivalent(A::UncertainValue, B::UncertainValue, k=1.0) =
    abs(A.value-B.value) <= k * sqrt(A.sigma^2+B.sigma^2)

uv(val::Real, sigma::Real) = UncertainValue(val, sigma)

function parse(::Type{UncertainValue}, str::AbstractString)::UncertainValue
    sp=split(str, r"(?:\+\-|\-\+|±)" )
    if length(sp)>=1
        val=parse(Float64, sp[1])
        sigma= length(sp)>=2 ? parse(Float64, sp[2]) : 0.0
    else
        error("Unable to parse " + str + " as an uncertain value.")
    end
    UncertainValue(val, sigma)
end

Base.show(io::IO,  uv::UncertainValue) =
    print(io, "$(uv.value) ± $(uv.sigma)")


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
function checkcovariance!(cov::AbstractMatrix{Float64}, tol=1.0e-12)::Bool
    sz = size(cov)
    if length(sz) ≠ 2
        error("The covariance must be a matrix.")
    end
    if sz[1] ≠ sz[2]
        error("The covariance matrix must be square.")
    end
    for rc in 1:sz[1]
        if cov[rc,rc] < 0.0
            error("The diagonal elements must all be non-negative. -> ",cov[rc,rc])
        end
    end
    for r in 1:sz[1]
        for c in 1:r-1
            if !isapprox(cov[r,c], cov[c,r], atol=tol*sqrt(cov[r,r]*cov[c,c]))
                error("The variances must be symmetric. -> ",cov[r,c]," ≠ ", cov[c,r])
            end
            cov[c,r]=cov[r,c] # Now force it precisely...
        end
    end
    for r in 1:sz[1]
        for c in 1:r-1
            cc = cov[r,c] / sqrt(cov[r,r]*cov[c,c])
            if abs(cc) > 1.0 + 1.0e-12
                error("The variances must have a correlation coefficient between -1.0 and 1.0 -> ",cc)
            end
            if abs(cc)>1.0
                cc = max(-1.0, min(1.0, cc))
                cov[r,c] = (cov[c,r] = cc * sqrt(cov[r,r]*cov[c,c]))
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
function checkcovariance!(cov::SparseMatrixCSC{Float64,Int}, tol=1.0e-12)::Bool
    # The generic AbstractMatrix implementation can be really slow on large sparce matrices.
    sz = size(cov)
    if length(sz) ≠ 2
        error("The covariance must be a matrix.")
    end
    if sz[1] ≠ sz[2]
        error("The covariance matrix must be square.")
    end
    for rc in 1:sz[1]
        if cov[rc,rc] < 0.0
            error("The diagonal elements must all be non-negative. (S) -> ",cov[rc,rc])
        end
    end
    for ci in findall(!iszero,cov)
        r,c = ci[1], ci[2]
        if !isapprox(cov[ci],cov[c,r], atol=abs(tol)*sqrt(cov[r,r]*cov[c,c]))
            error("The variances must be symmetric. (S) -> ",cov[ci] ," ≠ ", cov[c,r])
        end
        cov[c,r] = cov[ci] # Now force it precisely...
    end
    for ci in findall(!iszero,cov)
        r,c = ci[1], ci[2]
        cc = cov[ci] / sqrt(cov[r,r]*cov[c,c])
        if abs(cc) > 1.0 + 1.0e-12
            error("The variances must have a correlation coefficient between -1.0 and 1.0 (S) -> ",cc)
        end
        if abs(cc)>1.0
            cc = max(-1.0, min(1.0, cc))
            cov[ci] = (cov[c,r] = cc * sqrt(cov[r,r]*cov[c,c]))
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
    labels::Dict{Symbol,Int}
    values::AbstractVector{Float64}
    covariance::AbstractMatrix{Float64}
    UncertainValues(labels::Dict{Symbol, Int}, values::AbstractVector{Float64}, covar::AbstractMatrix{Float64}) =
        checkUVS!(labels, values, covar) ? new(labels, values, covar) : error("???")
end

uvs(labels::AbstractVector{Symbol}, values::AbstractVector{Float64}, covar::AbstractMatrix{Float64}) =
    UncertainValues(Dict{Symbol,Int}( [ (l, i) for (i, l) in enumerate(labels) ] ), values, covar)

function checkUVS!(labels::Dict{Symbol, Int}, values::AbstractVector{Float64}, covar::AbstractMatrix{Float64})
    if length(labels) ≠ length(values)
        error("The number of labels does not match the number of values.")
    end
    if length(labels) ≠ size(covar)[1]
        error("The number of labels does not match the dimension of the covariance matrix.")
    end
    checkcovariance!(covar)
end

σ(lbl::Symbol, uvs::UncertainValues) = sqrt(variance(lbl, uvs))

"""
    extract(uvs::UncertainValues, labels::Vector{Symbol})::Matrix
Extract the covariance matrix associated with the variables specified in labels
into a Matrix.
"""
function extract(uvs::UncertainValues, labels::Vector{Symbol})::Matrix
    m = zeros(length(labels),length(labels))
    for (r,rl) in enumerate(labels)
        for (c,cl) in enumerate(labels)
            m[r,c]=covariance(rl, cl, uvs)
        end
    end
    return m
end

Base.:*(aa::AbstractMatrix{Float64}, uvs::UncertainValues) =
    UncertainValues(uvs.labels, aa*uvs.values, aa*uvs.covariance*transpose(aa))

Base.:*(aa::Diagonal{Float64}, uvs::UncertainValues) =
    UncertainValues(uvs.labels, aa*uvs.values, aa*uvs.covariance*aa)

"""
    cat(uvss::AbstractArray{UncertainValues})::UncertainValues
Combines the disjoint UncertainValues in uvss into a single UncertainValues object.
"""
function Base.cat(uvss::AbstractArray{UncertainValues})::UncertainValues
    all = Dict{Symbol,Int}()
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
                covar[ridx, cidx] = covariance(rlbl,clbl,uvs)
            end
        end
    end
    checkcovariance!(covar)
    UncertainValues(all, values, covar)
end

function Base.show(io::IO, uvs::UncertainValues)
    trim(str, len) = str[1:min(len,length(str))]*" "^max(0,len-min(len,length(str)))

    lbls = labels(uvs)
    print(io, "Symbol    ")
    print(io, "    Value    ")
    print(io, "          ")
    for l in lbls
        print(io, trim(repr(l),10))
        print(io, "  ")
    end
    println(io)

    for (r, rl) in enumerate(lbls)
        print(io, trim(repr(rl),10))
        print(io, @sprintf(" | %-8.3g |", value(rl, uvs)))
        print(io, r==length(uvs.labels)[1]/2 ? "  ±  |" : "     |")
        for cl in lbls
            print(io, @sprintf("   %-8.3g ", covariance(rl, cl, uvs)))
        end
        println(io," |")
    end
end

"""
    labels(uvs::UncertainValues)
A alphabetically sorted list of the labels
"""
labels(uvs::UncertainValues) = sort( [ keys(uvs.labels)...], lt= (l,m) -> isless(repr(l),repr(m)))

function Base.getindex(uvs::UncertainValues, lbl::Symbol)::UncertainValue
    idx = uvs.labels[lbl]
    UncertainValue(uvs.values[idx], sqrt(uvs.covariance[idx,idx]))
end

function Base.get(uvs::UncertainValues, lbl::Symbol, def::UncertainValue)::UncertainValue
    idx = get(uvs.labels, lbl, -1)
    return idx ≠ -1 ? UncertainValue(uvs.values[idx], sqrt(uvs.covariance[idx,idx])) : def
end

Base.length(uvs::UncertainValues) = length(uvs.labels)
Base.size(uvs::UncertainValues) = size(uvs.values)

"""
    value(lbl::Symbol, uvs::UncertainValues)
The value associate with the Symbol.
"""
value(lbl::Symbol, uvs::UncertainValues) =
    uvs.values[uvs.labels[lbl]]

"""
    values(uvs::UncertainValues)
A Dict containing Symbol => UncertainValue for each row in uvs.
"""
Base.values(uvs::UncertainValues) =
    Dict( (lbl, uvs[lbl]) for lbl in keys(uvs.labels))

"""
   covariance(lbl1::Symbol, lbl2::Symbol, uvs::UncertainValues)
The covariance between the two variables.
"""
covariance(lbl1::Symbol, lbl2::Symbol, uvs::UncertainValues) =
    uvs.covariance[uvs.labels[lbl1], uvs.labels[lbl2]]

"""
   variance(lbl::Symbol, uvs::UncertainValues)
The variance associated with the specified Symbol.
"""
variance(lbl::Symbol, uvs::UncertainValues) =
    uvs.covariance[uvs.labels[lbl], uvs.labels[lbl]]

"""
    uncertainty(lbl::Symbol, uvs::UncertainValues, k::Float64=1.0)
The uncertainty associated with specified label (k σ where default k=1)
"""
uncertainty(lbl::Symbol, uvs::UncertainValues, k::Float64=1.0) =
    k*sqrt(uvs.covariance[uvs.labels[lbl], uvs.labels[lbl]])

struct Jacobian
    entries::AbstractMatrix{Float64}
    inputs::Dict{Symbol,Int}
    outputs::Dict{Symbol,Int}
    Jacobian(input::AbstractArray{Symbol}, output::AbstractArray{Symbol}, entries::AbstractMatrix{Float64}) =
        (size(entries)[2] == length(input)) && (size(entries)[1] == length(output)) ?
          new(entries, buildDict(input), buildDict(output)) : error("The output and input lengths must match the row and column dimensions of the matrix.")
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
            res[:,c]=jac.entries[:,jac.inputs[l]]
        end
        res
    end
    j=extract(jac,uvs)
    j*uvs.covariance*transpose(j)
end
