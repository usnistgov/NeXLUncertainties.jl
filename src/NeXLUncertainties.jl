module NeXLUncertainties

using Requires

include("uncertainValue.jl")
export UncertainValue # A simple value with uncertainty
export uv # Constructs an UncertainValue from values v,σ
export poisson # Creates an UncertainValue from a count statistic limited integer
export multiply # multiply UncertainValue structs
export divide # divide UncertainValue structs
export add
export variance # σ^2
export equivalent # Check whether two UncertainValue are equivalent values
export uncertainty # k-σ uncertainty
export σ # Sigma
export value # Value portion

include("label.jl")
export Label # Abstract type to label variables in UncertainValues object
export label # Construct a Label from a struct
export labelsByType # filter a vector of labels for the specified Label type(s)
export @nl_str # Shortcut to create a label from a string

# Apply labels to the values in an array
include("labeledvalues.jl")
export LabeledValues
export toarray, fromarray

include("uncertainValues.jl")
export UncertainValues # An array of values with covariances
export uvs # Constructs an UncertainValues object
export covariance # Extract the covariance associated with the specified label
export checkcovariance! # Ensure that the covariance matrix is valid
export labels # The labels for the values in an UncertainValues object
export cat # Combine UncertainValues objects into a single UncertainValues object
export σ # Sigma
export asa # Like an imperfect convert(...)
export fractional # fractional uncertainty
export correlation # Pearson correlation coefficient
export estimated # Estimate an UncertainValues from an ensemble of measurements.
export labeledvalues # Extract the values as a LabeledValues object
export extract #

include("propagate.jl")
export MeasurementModel
export propagate
export mcpropagate
export compute
export MaintainInputs
export AllInputs
export ParallelMeasurementModel
export parallel
export ComposedMeasurementModel
export MMResult

function __init__()
    @require DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0" include("dataframes.jl")
    @require LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f" include("latexstrings.jl")
end

end
