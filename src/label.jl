"""
`Label` is the root abstract type used to provide unique identifiers for variables within
`UncertainValues` and `LabeledValues`. See [BasicLabel]
"""


abstract type Label end


"""
`BasicLabel{T}` is a mechanism for creating `Label` around other objects.
For example, BasicLabel{String} would create a `Label` using a `String`
to provide a unique identifier.
"""
struct BasicLabel{T} <: Label
    value::T
    hc::UInt # cache the hash to ensure Dict and Set searches are fast

    BasicLabel(v::T) where {T} = new{T}(v, hash(v, UInt(0xEA7F00DBADF00D)))
end

label(item::T) where {T} = BasicLabel(item)

macro nl_str(str::AbstractString)
    BasicLabel(str)
end

Base.show(io::IO, bl::BasicLabel) = print(io, "$(bl.value)")
Base.isequal(bl1::BasicLabel{T}, bl2::BasicLabel{T}) where {T} =
    isequal(bl1.hc, bl2.hc) && isequal(bl1.value, bl2.value)
Base.isequal(l1::Label, l2::Label) = false
Base.hash(bl::BasicLabel, h::UInt) = hash(bl.hc, h)


"""
labelsByType(ty::Type{<:Label}, labels::AbstractVector{<:Label})
labelsByType(types::AbstractVector{DataType}, labels::AbstractVector{<:Label})

Extracts all of a specific type of `Label` from a list of `Label`s. The second version
extracts multiple different types in a single call.
"""
labelsByType(ty::Type{<:Label}, labels::AbstractVector{<:Label}) =
    filter(lbl -> typeof(lbl) == ty, labels)

labelsByType(types::AbstractVector{DataType}, labels::AbstractVector{<:Label}) =
    mapreduce(ty -> labelsByType(ty, labels), append!, types)
