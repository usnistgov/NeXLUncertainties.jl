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
    index::Dict{<:Label,Int}

    function LabeledValues(labels::AbstractVector{<:Label}, values::AbstractVector{Float64})
        @assert length(unique(labels)) == length(labels) "The labels [$labels] are not all unique."
        index = Dict{Label,Int}(labels[i] => i for i in eachindex(labels))
        return new(copy(values), index)
    end
    function LabeledValues(
        labels::NTuple{N,Label},
        values::NTuple{N,Float64},
    ) where {N<:Any}
        @assert length(unique(labels)) == length(labels) "The labels are not all unique."
        index = Dict{Label,Int}(labels[i] => i for i in eachindex(labels))
        return new([values...], index)
    end
    function LabeledValues(prs::Pair{<:Label,Float64}...)
        labels = [pr.first for pr in prs]
        values = [pr.second for pr in prs]
        return LabeledValues(labels, values)
    end
end

Base.eachindex(lv::LabeledValues) = eachindex(lv.values)

Base.show(io::IO, lv::LabeledValues) = print(
    io,
    "LabeledValues[\n" *
    join(("\t" * repr(lbl) * " => " * repr(lv[lbl]) for lbl in labels(lv)), "\n") *
    "\n]",
)

Base.copy(lv::LabeledValues) = LabeledValues(copy(lv.values), copy(lv.index))

Base.merge(lvs::LabeledValues...) =
    LabeledValues(vcat(map(labels, lvs)...), vcat(map(values, lvs)...))

"""
    value(lv::LabeledValues, lbl::Label)::Float64

Returns the value associated with the specified `Label`.
"""
value(lv::LabeledValues, lbl::Label)::Float64 = lv.values[indexin(lv, lbl)]

Base.length(lv::LabeledValues) = length(lv.values)
Base.getindex(lv::LabeledValues, lbl::Label)::Float64 = lv.values[indexin(lv, lbl)]
function Base.get(lv::LabeledValues, lbl::Label, def)
    idx = get(lv.index, lbl, nothing)
    return isnothing(idx) ? def : lv.values[idx]
end

function Base.setindex!(lv::LabeledValues, val::Real, lbl::Label)
    @assert !Base.haskey(lv.index, lbl) "$lbl already exists in LabeledValues - $lv"
    push!(lv.values, val)
    lv.index[lbl] = length(lv.values)
end

"""
    indexin(lv::LabeledValues, lbl::Label)::Int

Returns the index associated with the specified `Label`.
"""
Base.indexin(lv::LabeledValues, lbl::Label)::Int = lv.index[lbl]

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

A copy `Vector` of the values in the same order as `labels(lv)`.
"""
Base.values(lv::LabeledValues)::Vector{Float64} = copy(lv.values)


"""
    toarray(lv::LabeledValues)

`toarray(...)` and `fromarray(...)` are a pair of functions for making `LabeledValues`
compatible with functions that take as arguments a vector of Float64.  This is common
for non-linear optimization functions.  `toarray(lv)` takes a `LabeledValues` and returns
an ordered vector of Float64.  `fromarray(f,lv)` is a little more subtle as it returns
a function `g` such that `g(toarray(lv)) = f(lv)`.  That is, `fromarray(...)` allows you
to use `f(lv)` as an argument to functions that expect a vector argument.
"""
toarray(lv::LabeledValues) = values(lv)

"""
    fromarray(f::Function, lv::LabeledValues)

`toarray(...)` and `fromarray(...)` are a pair of functions for making `LabeledValues`
compatible with functions that take as arguments a vector of Float64.  This is common
for non-linear optimization functions.  `toarray(lv)` takes a `LabeledValues` and returns
an ordered vector of Float64.  `fromarray(f,lv)` is a little more subtle as it returns
a function `g` such that `g(toarray(lv)) = f(lv)`.  That is, `fromarray(...)` allows you
to use `f(lv)` as an argument to functions that expect a vector argument.
"""
function fromarray(f::Function, lv::LabeledValues, constants::LabeledValues=nothing)
    return if isnothing(constants)
        v->f(LabeledValues(labels(lv), v))
    else
        v->f(merge(LabeledValues(labels(lv), v), constants))
    end
end
    

