struct Jacobian
    entries::AbstractMatrix{Float64, 2}
    inputs::Dict{<:Label,Int}
    outputs::Dict{<:Label,Int}
    function Jacobian(
        input::AbstractArray{<:Label},
        output::AbstractArray{<:Label},
        entries::AbstractMatrix{Float64, 2},
    )
        @assert size(entries,1)==length(output) && size(entries,2)==length(input)
            "The output and input lengths must match the row and column dimensions of the matrix."
        return new(entries, buildDict(input), buildDict(output))
    end
end

inputLabels(jac::Jacobian) = keys(jac.inputs)

outputLabels(jac::Jacobian) = keys(jac.outputs)


"""
    jacobian(input::UncertainValues, outputLabels::AbstractArray{<:Label})

Builds a Jacobian ordered to mate with the `input`::UncertainValues object and the specified
set of output labels.  `input` defines the columns and `outputLabels` defines the rows.
"""
function jacobian(input::UncertainValues, outputLabels::AbstractArray{<:Label})
    inputLabels = naturalorder(input)
    Jacobian(inputLabels, outputLabels, zeros(Float64, length(outputLabels), length(inputLabels)))
end
"""
    propagate(jac::Jacobian, uvs::UncertainValues)::Matrix

Propagate the covariance matrix in uvs using the specified Jacobian creating a new covariance matrix.
C' = J⋅C⋅transpose(J)
"""
function propagate(jac::Jacobian, uvs::UncertainValues)::Matrix
    function extract(jac::Jacobian, uvs::UncertainValues)::Matrix
        res = zeros(length(jac.entries))
        jac.inputs

        for (l, c) in uvs.labels
            res[:, c] = jac.entries[:, jac.inputs[l]]
        end
        res
    end
    j = extract(jac, uvs)
    return j * uvs.covariance * transpose(j)
end
