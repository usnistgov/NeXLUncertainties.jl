using Distributions
using Random
using LinearAlgebra
using Base.Threads
using Base.Iterators

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

    function LabeledValues(
        labels::Vector{<:Label},
        values::Vector{Float64}
    )
        forward = Dict{Label,Float64}( labels[i]=>values[i] for i in eachindex(labels))
        reverse = Dict{Label, Int}( labels[i] => i for i in eachindex(labels))
        return new(forward, reverse)
    end

    function LabeledValues(uvs::UncertainValues)
        forward = Dict{Label, Float64}( lbl => uvs.values[i] for (lbl, i) in uvs.labels )
        return new(forward, uvs.labels)
    end
end

"""
    value(lv::LabeledValues, lbl::Label)::Float64

Returns the value associated with the specified `Label`.
"""
value(lv::LabeledValues, lbl::Label)::Float64 =
    lv.forward[lbl]

Base.length(lv::LabeledValues) = length(lv.forward)

Base.getindex(lv::LabeledValues, lbl::Label)::Float64 =
    lv.forward[lbl]

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
haskey(lv::LabeledValues, lbl::Label) =
    haskey(lv.forward, lbl)

keys(lv::LabeledValues) = keys(lv.forward)

"""
    labels(lv::LabeledValues)::Vector{Label}

A list of the `Label`s in order.
"""
function labels(lv::LabeledValues)::Vector{Label}
    res = Array{Label}(undef, length(lv.reverse))
    for (lbl, idx) in lv.reverse
        res[idx] = lbl
    end
    return res
end

"""
    MeasurementModel

MeasurementModel is an abstract type representing an explicit measurement model.
MeasurementModels take a LabeledValues of input values and compute a Tuple consisting
of a Vector of Label(s) for the output values, a Vector of output Float64 and
a Jacobian matrix.

The Jacobian has rows defined by the AbstractVector{Label} in the return Tuple and
columns defined by labels(inputs).  Each element in the Jacobian represents the
partial derivative of an output value (row) with respect to an input value (column).

    compute(mm::MeasurementModel, inputs::LabeledValues, withJac::Boolean)::MMReturn

where

    MMReturn = Tuple{AbstractVector{Label}, AbstractVector{Float64}, Union{Missing,AbstractMatrix{Float64}}}
"""
abstract type MeasurementModel end

const MMReturn = Tuple{AbstractVector{Label}, AbstractVector{Float64}, Union{Missing,AbstractMatrix{Float64}}}

"""
    propagate(mm::MeasurementModel, uvs::UncertainValues)::UncertainValues

Propagate the input (measured values as UncertainValues) through the MeasurementModel to
produce the output values (as UncertainValues).
"""
function propagate(mm::MeasurementModel, inputs::UncertainValues)::UncertainValues
    (outlabels, res, jac) = compute(mm, LabeledValues(inputs), true)
    return uvs(outlabels, res, jac*inputs.covariance*transpose(jac))
end

function mcpropagate(mm::MeasurementModel, inputs::UncertainValues, n::Int, rng::AbstractRNG = Random.GLOBAL_RNG)::UncertainValues
    mvn = MvNormal(inputs.values, inputs.covariance)
    (outlabels, vals, _) = compute(mm, LabeledValues(inputs), false)
    println(outlabels)
    samples, inlabels = Array{Float64}(undef, (length(outlabels), n)), labels(inputs)
    for i in 1:n
        inputs = LabeledValues(inlabels, rand(rng, mvn))
        (outlabels, res, jac) = compute(mm, inputs, false)
        samples[:, i] = res
    end
    f = fit(MvNormal, samples)
    return uvs(outlabels, f.μ, f.Σ.mat)
end

"""
    compute(mm::MeasurementModel, uvs::LabeledValues)::LabeledValues

Calculate the output values for the specified set of input LabeledValues.
"""
function compute(mm::MeasurementModel, inputs::LabeledValues)::LabeledValues
    (outlabels, res, jac) = compute(mm, inputs, false)
    return LabeledValues(outlabels, res)
end

"""
    CarryOver

Carry over a subset of the inputs to the next step in a calculation.  Typically used in
consort with a SerialMeasurementModel when inputs to this step will also be required
in subsequent steps.
"""
struct CarryOver <: MeasurementModel
    labels::Vector{Label}

    CarryOver(uvs::UncertainValues) = new(naturalorder(uvs))
    CarryOver(labels::AbstractVector{Label}) = new(labels)
end

function compute(mm::CarryOver, inputs::LabeledValues, withJac::Bool)::MMReturn
    res = collect(map(lbl->value(inputs,lbl), mm.labels))
    jac = withJac ? Matrix{Float64}(I, length(mm.labels), length(mm.labels)) : missing
    return (copy(mm.labels), res, jac)
end

struct ParallelMeasurementModel <: MeasurementModel
    models::Vector{MeasurementModel}
    multi::Bool

    ParallelMeasurementModel(models::AbstractVector{MeasurementModel}, multithread=false) =
        new(models, multithread)
end

function compute(mm::ParallelMeasurementModel, inputs::LabeledValues, withJac::Bool)::MMReturn
    tmp = Vector{Tuple}(undef, length(mm.models))
    # This can be readily threaded using Threads.@thread
    if mm.multi
        @threads for i in eachindex(mm.models)
            tmp[i] = compute(mm.models[i], inputs, withJac)
        end
    else
        for i in eachindex(mm.models)
            tmp[i] = compute(mm.models[i], inputs, withJac)
        end
    end
    lbls = reduce(append!, tmp[i][1] for i in eachindex(tmp))
    res = reduce(append!, tmp[i][2] for i in eachindex(tmp))
    jacs = [ tmp[i][3] for i in eachindex(tmp) ]
    jac = withJac ? vcat(jacs...) : missing
    return (lbls, res, jac)
end

struct SerialMeasurementModel <: MeasurementModel
    models::Vector{MeasurementModel}
end

function compute(mm::SerialMeasurementModel, inputs::LabeledValues, withJac::Bool)::MMReturn
    current, currjac = inputs, missing
    for model in models
        (lbls, res, jac) = compute(model, current)
        current = LabeledValues(lbls,res)
        if withJac
            currjac = ismissing(currjac) ? jac : jac*currjac
        end
    end
    return (labels(current), values(current), currjac)
end
