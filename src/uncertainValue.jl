using Printf
using Statistics
using LaTeXStrings

"""
    UncertainValue

Represents a floating point numerical value and an associated uncertainty (1σ).
"""
struct UncertainValue <: AbstractFloat
    value::Float64
    sigma::Float64

    function UncertainValue(val::Real, sigma::Real)
        @assert sigma >= 0.0
        return new(convert(Float64, val), convert(Float64, sigma))
    end
end

Base.convert(::Type{UncertainValue}, n::Real) = UncertainValue(convert(Float64, n), 0.0)
Base.convert(::Type{UncertainValue}, n::UncertainValue) = n

Base.zero(::Type{UncertainValue}) = UncertainValue(0.0,0.0)
Base.one(::Type{UncertainValue}) = UncertainValue(1.0,0.0)
Base.add_sum(x::UncertainValue, y::UncertainValue)::Real = x.value + y.value
Base.add_sum(x::AbstractFloat, y::UncertainValue)::Real = x + y.value

"""
    Base.round(uva::UncertainValue; defdigits=4)

Provides a (somewhat) intelligent formatter for `UncertainValue` objects that uses the
uncertainty to suggest the number of digits to display.
"""
function Base.round(uva::UncertainValue; defdigits=4)
    r = (σ(uva)>=1.0e-10*value(uva)) && (abs(value(uva))>1.0e-100) ? convert(Int, ceil(max(1,log10(max(σ(uva), abs(value(uva)))/σ(uva)))))+2 : defdigits
    return uv(round(value(uva), RoundNearestTiesUp, sigdigits=r), round(σ(uva), RoundNearestTiesUp, sigdigits=2))
end



"""
    poisson(val::Int)

Creates an UncertainValue from an integer which is assumed to have σ = sqrt(val).
"""
poisson(val::Int) = UncertainValue(convert(Float64, val), sqrt(convert(Float64, val)))

"""
    uv(val::Real, σ::Real)

Create an UncertainValue from a real value and 1σ uncertainty.
"""
uv(val::Real, σ::Real) = UncertainValue(val, σ)


"""
    max(uv1::UncertainValue, uv2::UncertainValue)

Determines the maximum of two values by comparing first the value and second the
uncertainty. (Equal value defer to the larger uncertainty being considered the 'max'.)

If `value(uv1)==value(uv2)` then it is undefined which is larger but to ensure a constant
ordering I define the one with a larger uncertainty to be larger since it is more probable
that it could be larger.
"""
Base.max(uv1::UncertainValue, uv2::UncertainValue) =
    uv1.value ≠ uv2.value ? (uv1.value > uv2.value ? uv1 : uv2) : (uv1.sigma > uv2.sigma ? uv1 : uv2)

"""
    min(uv1::UncertainValue, uv2::UncertainValue)

Determines the maximum of two values by comparing first the value and second the
uncertainty. (Equal value defer to the larger uncertainty being considered the 'min'.)

If `value(uv1)==value(uv2)` then it is undefined which is smaller but to ensure a constant
ordering I define the one with a larger uncertainty to be smaller since it is more probable
that it could be smaller.
"""
Base.min(uv1::UncertainValue, uv2::UncertainValue) =
    uv1.value ≠ uv2.value ? (uv1.value > uv2.value ? uv2 : uv1) : (uv1.sigma > uv2.sigma ? uv1 : uv2)

Base.minimum(uvs::AbstractVector{UncertainValue}) =
    reduce(min, uvs)

Base.minimum(xs::Tuple{UncertainValue, Vararg{UncertainValue}}) =
    reduce(min, xs)

Base.maximum(uvs::AbstractVector{UncertainValue}) =
    reduce(max, uvs)

Base.maximum(xs::Tuple{UncertainValue, Vararg{UncertainValue}}) =
    reduce(max, xs)

Base.sum(xs::UncertainValue...) =
    UncertainValue(sum(value(x) for x in xs), sqrt(sum(variance(x) for x in xs)))

Base.sum(uvs::AbstractVector{UncertainValue}) =
    UncertainValue(sum(value(x) for x in uvs), sqrt(sum(variance(x) for x in uvs)))


"""
    Statistics.mean(uvs::UncertainValue...)
    Statistics.mean(uvs::AbstractVector{UncertainValue})

The variance weighted mean of a collection of UncertainValue items.
(see https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Variance_weights).
"""
Statistics.mean(uvs::AbstractVector{UncertainValue}) =
    UncertainValue(
        sum(value(x)/variance(x) for x in uvs)/sum(1.0/variance(x) for x in uvs),
        sqrt(1.0/sum(1.0/variance(x) for x in uvs)))
Statistics.mean(uvs::UncertainValue...) =
    UncertainValue(
        sum(value(x)/variance(x) for x in uvs)/sum(1.0/variance(x) for x in uvs),
        sqrt(1.0/sum(1.0/variance(x) for x in uvs)))

"""
    Statistics.std(uvs::UncertainValue...)

The standard deviation of the value portion of the collection of UncertainValue items.
"""
Statistics.std(uvs::UncertainValue...) =
    std(value(x) for x in uvs)
Statistics.std(uvs::AbstractVector{UncertainValue}) =
    std(value(x) for x in uvs)

"""
    variance(uv::UncertainValue)

Returns σ².
"""
variance(uv::UncertainValue) = uv.sigma^2


"""
    variance(f::Real)

Returns 0.
"""
variance(f::Real) = zero(typeof(f))

Base.isless(uv1::UncertainValue, uv2::UncertainValue) =
    isequal(uv1.value, uv2.value) ? !isless(uv1.sigma, uv2.sigma) : isless(uv1.value, uv2.value)

"""
    σ(uv::UncertainValue)

Returns the 1-σ uncertainty)
"""
σ(uv::UncertainValue) = uv.sigma


"""
    σ(r::Real)

Returns 0.
"""
σ(f::Real) = zero(typeof(f))
σ(m::Missing) = m

"""
    uncertainty(uv::UncertainValue, k::Real=1.0)

Returns the k-σ uncertainty (defaults to k=1.0)
"""
uncertainty(uv::UncertainValue, k::Real=1) = k*uv.sigma

"""
    uncertainty(f::Real, k::Real=1.0)

Returns 0.0.
"""
uncertainty(f::Real, k::Real=1) = zero(typeof(f))

"""
    fractional(uv::UncertainValue)

Computes the fractional uncertainty.
"""
fractional(uv::UncertainValue)::Float64 = abs(uv.value) > 1.0e-15*abs(uv.sigma) ? abs(uv.sigma/uv.value) : Inf64

"""
    fractional(f::Real)

Returns 0
"""
fractional(f::Real) = zero(typeof(f))

"""
    value(uv::UncertainValue)

Returns the value portion. (uv.value)
"""
value(uv::UncertainValue) = uv.value

"""
    value(f::Real)

Returns f
"""
value(f::Real) = f
value(m::Missing) = m

"""
    pearson(uv1::UncertainValue, uv2::UncertainValue, covar::Real)

Computes the Pearson correlation coefficient given the covariance between two UncertainValue.
"""
function pearson(uv1::UncertainValue, uv2::UncertainValue, covar::Real)
    res = covar / (σ(uv1) * σ(uv2))
    @assert (res>=-1.0) & (res<=1.0) "The Pearson correlation coefficient must be on [-1.0, 1.0] - $res"
    return res
end

"""
    covariance(uv1::UncertainValue, uv2::UncertainValue, correlation::Real)

Computes the covariance given the correlation coefficient between two UncertainValue.
"""
function covariance(uv1::UncertainValue, uv2::UncertainValue, correlation::Real)
    @assert (correlation>=-1.0) & (correlation<=1.0) "The Pearson correlation coefficient must be on [-1.0, 1.0] - $correlation"
    return correlation * σ(uv1) * σ(uv2)
end

"""
    divide(n::UncertainValue, d::UncertainValue, cc::AbstractFloat)

Computes `n/d` where `n` and `d` are UncertainValue and `cc` is the correlation coefficient
defined as `cc = covar(n,d)/( σ(n), σ(d) )`
"""
function divide(n::UncertainValue, d::UncertainValue, cc::AbstractFloat)
    @assert (cc>=-1.0) & (cc<=1.0) "The Pearson correlation coefficient must be on [-1.0, 1.0] - $cc"
    f, sab = n.value/d.value, covariance(n, d, cc)
    return UncertainValue(f, sqrt(f^2*((n.sigma/n.value)^2+(d.sigma/d.value)^2 - (2.0*sab)/(n.value*d.value))))
end

"""
    multiply(a::UncertainValue, b::UncertainValue, cc::AbstractFloat)

Computes `a*b` where `a` and `b` are UncertainValue and `cc` is the correlation coefficient
defined as `cc = covar(a,b)/( σ(a), σ(b) )`
"""
function multiply(a::UncertainValue, b::UncertainValue, cc::AbstractFloat)
    @assert (cc>=-1.0) & (cc<=1.0) "The Pearson correlation coefficient must be on [-1.0, 1.0] - $cc"
    f, sab = a.value*b.value, covariance(a, b, cc)
    return UncertainValue(f, sqrt(f^2*((a.sigma/a.value)^2+(b.sigma/b.value)^2 + (2.0*sab)/(a.value*b.value))))
end

"""
    add(ka::Real, a::UncertainValue, kb::Real, b::UncertainValue, cc::AbstractFloat)

Computes `ka*a + kb*b` where `a` and `b` are UncertainValue and `cc` is the correlation
coefficient defined as `cc = covar(a,b)/( σ(a), σ(b) )`
"""
add(ka::Real, a::UncertainValue, kb::Real, b::UncertainValue, cc::AbstractFloat) =
    UncertainValue(convert(Float64, ka)*a.value+convert(Float64, kb)*b.value, sqrt((ka*a.sigma)^2+(kb*b.sigma)^2 + 2.0*ka*kb*covariance(a,b,cc)))

Base.:*(a::Real, B::UncertainValue) =
    UncertainValue(convert(Float64,a)*B.value, abs(convert(Float64,a)*B.sigma))
Base.:*(A::UncertainValue, b::Real) =
    UncertainValue(convert(Float64,b)*A.value, abs(convert(Float64,b)*A.sigma))

Base.:-(A::UncertainValue) =
    UncertainValue(-A.value,A.sigma)
Base.:+(A::UncertainValue) =
    UncertainValue(A.value,A.sigma)

Base.abs(A::UncertainValue) =
    A.value>=0.0 ? A : UncertainValue(abs(A.value),A.sigma)

Base.:/(a::Real, B::UncertainValue) =
    UncertainValue(convert(Float64,a)/B.value, abs((convert(Float64,a)*B.sigma)/(B.value*B.value)))
Base.:/(A::UncertainValue, b::Real) =
    UncertainValue(A.value/convert(Float64,b), abs(A.sigma/convert(Float64,b)))

Base.inv(B::UncertainValue) =
    UncertainValue(1.0/B.value, abs(B.sigma/(B.value^2)))

Base.:^(A::UncertainValue, b::Real) =
    UncertainValue(A.value^b, abs(A.value^(b-one(b)) * b * A.sigma))

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

function Base.parse(::Type{UncertainValue}, str::AbstractString)::UncertainValue
    sp=split(str, r"(?:\+\-|\-\+|±)" )
    if length(sp)>=1
        val=parse(Float64, sp[1])
        sigma= length(sp)>=2 ? parse(Float64, sp[2]) : 0.0
    else
        error("Unable to parse " + str + " as an uncertain value.")
    end
    UncertainValue(val, sigma)
end

function Base.show(io::IO,  uv::UncertainValue)
    lv, ls, lr = map(v->floor(Int,log10(abs(v))), ( uv.value, uv.sigma, uv.value/uv.sigma))
    # @show ( lv, ls, lr )
    if (ls < -4) || (ls>6)
        if lr < 1
            @printf(io,"%0.1e ± %0.1e", uv.value, uv.sigma)
        elseif lr < 2
            @printf(io,"%0.2e ± %0.1e", uv.value, uv.sigma)
        elseif lr < 3
            @printf(io,"%0.3e ± %0.1e", uv.value, uv.sigma)
        elseif lr < 4
            @printf(io,"%0.4e ± %0.1e", uv.value, uv.sigma)
        elseif lr < 5
            @printf(io,"%0.5e ± %0.1e", uv.value, uv.sigma)
        else
            @printf(io,"%0.6e ± %0.1e", uv.value, uv.sigma)
        end
    else # lv < 0 && lv > -4
        if lr > 6
            @printf(io, "%e ± %0.1e",uv.value,uv.sigma)
        elseif ls >= 0
            @printf(io,"%0.0f ± %0.0f",uv.value,uv.sigma)
        elseif ls==-1
            @printf(io,"%0.2f ± %0.2f",uv.value,uv.sigma)
        elseif ls==-2
            @printf(io,"%0.3f ± %0.3f",uv.value,uv.sigma)
        else# if ls==-3
            @printf(io,"%0.4f ± %0.4f",uv.value,uv.sigma)
        end
    end
end

"""
    function asa(::Type{LaTeXString},  uv::UncertainValue)

Converts an `UncertainValue` to a `LaTeXString` in a reasonable manner.
Requires `\\usepackage{siunitx}` which defines `\\num{}`.
"""
function asa(::Type{LaTeXString},  uv::UncertainValue)
    lv, ls, lr = map(v->floor(Int,log10(abs(v))), ( uv.value, uv.sigma, uv.value/uv.sigma))
    # @show ( lv, ls, lr )
    res = if (ls < -4) || (ls>6)
        if lr < 1
            @sprintf("\$\\num{%0.1e} \\pm \\num{%0.1e}\$", uv.value, uv.sigma)
        elseif lr < 2
            @sprintf("\$\\num{%0.2e} \\pm \\num{%0.1e}\$", uv.value, uv.sigma)
        elseif lr < 3
            @sprintf("\$\\num{%0.3e} \\pm \\num{%0.1e}\$", uv.value, uv.sigma)
        elseif lr < 4
            @sprintf("\$\\num{%0.4e} \\pm \\num{%0.1e}\$", uv.value, uv.sigma)
        elseif lr < 5
            @sprintf("\$\\num{%0.5e} \\pm \\num{%0.1e}\$", uv.value, uv.sigma)
        else
            @sprintf("\$\\num{%0.6e} \\pm \\num{%0.1e}\$", uv.value, uv.sigma)
        end
    else # lv < 0 && lv > -4
        if lr > 6
            @sprintf("\$\\num{%e} \\pm \\num{%0.1e}\$",uv.value,uv.sigma)
        elseif ls >= 0
            @sprintf("\$\\num{%0.0f} \\pm \\num{%0.0f}\$",uv.value,uv.sigma)
        elseif ls==-1
            @sprintf("\$\\num{%0.2f} \\pm \\num{%0.2f}\$",uv.value,uv.sigma)
        elseif ls==-2
            @sprintf("\$\\num{%0.3f} \\pm \\num{%0.3f}\$",uv.value,uv.sigma)
        else# if ls==-3
            @sprintf("\$\\num{%0.4f} \\pm \\num{%0.4f}\$",uv.value,uv.sigma)
        end
    end
    return latexstring(res)
end