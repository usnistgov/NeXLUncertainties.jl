using DataFrames

"""
    LabeledValues

A `LabeledValues` object represents an efficient way to deal with
arrays of `Label`ed values (`Float64`). Each value is indexed
via a `Label` so storing them in `Vector`s with be O(N) whereas
storing the `Label` and value in a `Dict` is O(1).  However, storing
them in a `Dict` loses the ordering which we would also like
to retain. `LabeledValues` are O(1) while retaining the order of
the labels.
"""
struct LabeledValues
    forward::Dict{<:Label, Float64}
    reverse::Dict{<:Label, Int}

    LabeledValues(
        forward::Dict{<:Label, Float64},
        reverse::Dict{<:Label, Int}
    ) = new(forward, reverse)

    function LabeledValues(
        labels::AbstractVector{<:Label},
        values::AbstractVector{Float64}
    )
        forward = Dict{Label,Float64}( labels[i]=>values[i] for i in eachindex(labels))
        reverse = Dict{Label, Int}( labels[i] => i for i in eachindex(labels))
        return new(forward, reverse)
    end
end

Base.show(io::IO, lv::LabeledValues) =
    print(io,"LabeledValues[\n"*join(("\t"*repr(lbl)*" => "*repr(val) for (lbl, val) in lv.forward), "\n")*"\n]")

Base.copy(lv::LabeledValues) = LabeledValues(copy(lv.forward), copy(lv.reverse))

Base.merge(lvs::LabeledValues...) =
    LabeledValues(vcat(map(labels, lvs)...), vcat(map(values, lvs)...))

"""
    value(lv::LabeledValues, lbl::Label)::Float64

Returns the value associated with the specified `Label`.
"""
value(lv::LabeledValues, lbl::Label)::Float64 =
    lv.forward[lbl]

Base.length(lv::LabeledValues) = length(lv.forward)

Base.getindex(lv::LabeledValues, lbl::Label)::Float64 =
    lv.forward[lbl]

function Base.setindex!(lv::LabeledValues, val::Real, lbl::Label)
    @assert !Base.haskey(lv.forward, lbl) "$lbl already exists in LabeledValues - $lv"
    lv.forward[lbl] = val
    lv.reverse[lbl] = length(lv.forward)
end

"""
    index(lv::LabeledValues, lbl::Label)::Int

Returns the index associated with the specified `Label`.
"""
Base.indexin(lbl::Label, lv::LabeledValues)::Int =
    lv.reverse[lbl]

"""
    haskey(lv::LabeledValues, lbl::Label)

Is there a value associated with the `Label`?
"""
Base.haskey(lv::LabeledValues, lbl::Label) =
    haskey(lv.forward, lbl)

Base.keys(lv::LabeledValues) = keys(lv.forward)

"""
    labels(lv::LabeledValues)::Vector{Label}

A `Vector` of the `Label`s in order.
"""
function labels(lv::LabeledValues)::Vector{Label}
    res = Array{Label}(undef, length(lv.reverse))
    for (lbl, idx) in lv.reverse
        res[idx] = lbl
    end
    return res
end

"""
    values(lv::LabeledValues)::Vector{Float64}

A `Vector` of the values in order.
"""
function Base.values(lv::LabeledValues)::Vector{Float64}
    res = Array{Float64}(undef, length(lv.reverse))
    for (lbl, idx) in lv.reverse
        res[idx] = lv.forward[lbl]
    end
    return res
end


function asa(::DataFrame, lv::LabeledValues)
    name, value = String[], Float64[]
    for lbl in labels(lv)
        push!(name, repr(lbl))
        push!(value, lv[lbl])
    end
    return DataFrame(Name=name, Value=value)
end
