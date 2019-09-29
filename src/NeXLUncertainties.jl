module NeXLUncertainties


include("uncertainty.jl")

export UncertainValue # A simple value with uncertainty
export multiply # multiply UncertainValue structs
export divide # divide UncertainValue structs
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
export value # Value portion
export extract # Extract a portion of the covariance matrix
export uv # Constructs an UncertainValue from values v,σ or string "v ± σ"
export Label # Abstract type to label variables in a UncertainValues object
export UncertainValues # An array of values with covariances
export covariance # Extract the covariance associated with the specified label
export checkcovariance! # Ensure that the covariance matrix is valid
export uvs # Constructs an UncertainValues object
export label # Construct a Label from a struct
export labels # The labels for the values in an UncertainValues object
export cat # Combine UncertainValues objects into a single UncertainValues object
export σ # Sigma
export fractional # fractional uncertainty
export bound # bound(min, max, val)

end
