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
    values::Vector{Float64}
    index::Dict{<:Label, Int}

    function LabeledValues(
        labels::AbstractVector{<:Label},
        values::AbstractVector{Float64}
    )
        @assert length(unique(labels)) == length(labels) "The labels are not all unique."
        index = Dict{Label, Int}( labels[i] => i for i in eachindex(labels))
        return new(copy(values), index)
    end
    function LabeledValues(
        labels::NTuple{N, Label},
        values::NTuple{N, Float64}
    ) where N <: Any
        @assert length(unique(labels)) == length(labels) "The labels are not all unique."
        index = Dict{Label, Int}( labels[i] => i for i in eachindex(labels))
        return new([values...], index)
    end
end

Base.eachindex(lv::LabeledValues) = eachindex(values)

Base.show(io::IO, lv::LabeledValues) =
    print(io,"LabeledValues[\n"*join(("\t"*repr(lbl)*" => "*repr(lv[lbl]) for lbl in labels(lv)), "\n")*"\n]")

Base.copy(lv::LabeledValues) = LabeledValues(copy(lv.values), copy(lv.index))

Base.merge(lvs::LabeledValues...) =
    LabeledValues(vcat(map(labels, lvs)...), vcat(map(values, lvs)...))

"""
    value(lv::LabeledValues, lbl::Label)::Float64

Returns the value associated with the specified `Label`.
"""
value(lv::LabeledValues, lbl::Label)::Float64 = lv.values[indexin(lbl, lv)]
Base.length(lv::LabeledValues) = length(lv.values)
Base.getindex(lv::LabeledValues, lbl::Label)::Float64 = lv.values[indexin(lbl,lv)]

function Base.setindex!(lv::LabeledValues, val::Real, lbl::Label)
    @assert !Base.haskey(lv.index, lbl) "$lbl already exists in LabeledValues - $lv"
    push!(lv.values,  val)
    index[lbl] = length(values) + 1
end

"""
    indexin(lbl::Label, lv::LabeledValues)::Int

Returns the index associated with the specified `Label`.
"""
Base.indexin(lbl::Label, lv::LabeledValues)::Int = lv.index[lbl]

Base.haskey(lv::LabeledValues, lbl::Label) = haskey(lv.index, lbl)
Base.keys(lv::LabeledValues) = keys(lv.index)

"""
    labels(lv::LabeledValues)::Vector{Label}

A `Vector` of the `Label`s in order.
"""
function labels(lv::LabeledValues)::Vector{Label}
    res = Array{Label}(undef, length(lv.index))
    for (lbl, idx) in lv.index
        res[idx] = lbl
    end
    return res
end

"""
    values(lv::LabeledValues)::Vector{Float64}

A copy `Vector` of the values in order.
"""
Base.values(lv::LabeledValues)::Vector{Float64} = copy(lv.values)

function asa(::DataFrame, lv::LabeledValues)
    name, value = String[], Float64[]
    for lbl in labels(lv)
        push!(name, repr(lbl))
        push!(value, lv[lbl])
    end
    return DataFrame(Name=name, Value=value)
end
