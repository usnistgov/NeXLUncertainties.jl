"""
    UncertainValue

Represents a floating point numerical value and an associated uncertainty (1σ).
"""
struct UncertainValue <: AbstractFloat
    value::Float64
    sigma::Float64
    UncertainValue(val::Real, sigma::Real) =
        convert(Float64, sigma) >= 0.0 ?
        new(convert(Float64, val), convert(Float64, sigma)) :
        error("σ must be non-negative.")
end

Base.convert(::Type{UncertainValue}, n::Real) =
    typeof(n) == UncertainValue ? n : UncertainValue(convert(Float64, n), 0.0)

Base.zero(::Type{UncertainValue}) = UncertainValue(0.0,0.0)
Base.one(::Type{UncertainValue}) = UncertainValue(1.0,0.0)

max(uv1::UncertainValue, uv2::UncertainValue) =
    uv1.value > uv2.value ? uv1 : uv2

min(uv1::UncertainValue, uv2::UncertainValue) =
    uv1.value < uv2.value ? uv1 : uv2

Base.sum(xs::Tuple{UncertainValue, Vararg{UncertainValue}}) =
    UncertainValue(sum(v->value(x) for x in xs), sqrt(sum(v->variance(x) for x in xs)))

variance(uv::UncertainValue) = uv.sigma^2
variance(f::Real) = 0.0

"""
    σ(uv::UncertainValue)

Returns the 1-σ uncertainty)
"""
σ(uv::UncertainValue) = uv.sigma
σ(f::Real) = 0.0

"""
    uncertainty(uv::UncertainValue, k::Real=1.0)

Returns the k-σ uncertainty (defaults to k=1.0)
"""
uncertainty(uv::UncertainValue, k::Real=1) = k*uv.sigma
uncertainty(f::Real, k::Real=1) = 0.0

"""
    fractional(uv::UncertainValue)

Computes the fractional uncertainty.
"""
fractional(uv::UncertainValue)::Float64 = uv.sigma/uv.value
fractional(f::Real) = 0.0

"""
    value(uv::UncertainValue)

Returns the value portion. (uv.value)
"""
value(uv::UncertainValue) = uv.value
value(f::Real) = f

multiply(a::Real, B::UncertainValue) =
    UncertainValue(convert(Float64,a)*B.value, abs(convert(Float64,a)*B.sigma))

multiply(A::UncertainValue, b::Real) =
    UncertainValue(convert(Float64,b)*A.value, abs(convert(Float64,b)*A.sigma))

function divide(n::UncertainValue, d::UncertainValue, cc::AbstractFloat)
    f, sab = n.value/d.value, sqrt(cc*n.sigma*d.sigma)
    return UncertainValue(f, sqrt(f^2*((n.sigma/n.value)^2+(d.sigma/d.value)^2 - (2.0*sab)/(n.value*d.value))))
end

function multiply(a::UncertainValue, b::UncertainValue, cc::AbstractFloat)
    f, sab = a.value*b.value, sqrt(cc*a.sigma*b.sigma)
    return UncertainValue(f, sqrt(f^2*((a.sigma/a.value)^2+(b.sigma/b.value)^2 + (2.0*sab)/(a.value*b.value))))
end

add(ka::AbstractFloat, a::UncertainValue, kb::AbstractFloat, b::UncertainValue, cc::AbstractFloat) =
    UncertainValue(ka*a.value+kb*b.value, sqrt((ka*a.sigma)^2+(kb*b.sigma)^2 + 2.0*ka*kb*sqrt(cc*a.sigma*b.sigma)))

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
    @printf(io,"%0.3g ± %0.3g",uv.value,uv.sigma)
