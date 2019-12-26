module NeXLUncertainties

include("uncertainValue.jl")
export UncertainValue # A simple value with uncertainty
export uv # Constructs an UncertainValue from values v,σ
export multiply # multiply UncertainValue structs
export divide # divide UncertainValue structs
export add
export exp # exponential function (e^X)
export log # log function
export sin # sine function
export cos # cosine function
export tan # tangent function
export sqrt # square root function
export power # power function (A^b)
export variance # σ^2
export isequal # Check UncertainValue structs for equality
export approx # Check UncertainValue structs for approximate quality
export equivalent # Check whether two UncertainValue are equivalent values
export uncertainty # k-σ uncertainty
export σ # Sigma
export value # Value portion

include("label.jl")
export Label # Abstract type to label variables in UncertainValues object
export label # Construct a Label from a struct
export @nl_str # Shortcut to create a label from a string

include("uncertainValues.jl")
export UncertainValues # An array of values with covariances
export uvs # Constructs an UncertainValues object
export extract # Extract a portion of the covariance matrix
export covariance # Extract the covariance associated with the specified label
export checkcovariance! # Ensure that the covariance matrix is valid
export labels # The labels for the values in an UncertainValues object
export cat # Combine UncertainValues objects into a single UncertainValues object
export σ # Sigma
export asa # Like an imperfect convert(...)
export fractional # fractional uncertainty

end
