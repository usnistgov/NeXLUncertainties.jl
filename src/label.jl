abstract type Label end

struct BasicLabel{T} <: Label
    value::T
    hc::UInt # cache the hash to ensure Dict and Set searches are fast

    BasicLabel(v::T) where {T} =
        new{T}(v, hash(v, UInt64(0xEA7F00DBADF00D)))
end

label(item::T) where {T} = BasicLabel(item)

macro nl_str(str::AbstractString)
    BasicLabel(str)
end

Base.show(io::IO, bl::BasicLabel) = print(io, "$(bl.value)")
Base.isequal(bl1::BasicLabel{T}, bl2::BasicLabel{T}) where {T} =
    isequal(bl1.hc,bl2.hc) && isequal(bl1.value, bl2.value)
Base.isequal(l1::Label, l2::Label) = false
Base.hash(bl::BasicLabel, h::UInt) =
    hash(bl.hc, h)
