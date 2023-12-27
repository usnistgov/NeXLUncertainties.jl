module NeXLUncertaintiesDataFramesExt

using DataFrames
using NeXLUncertainties

"""
    NeXLUncertainties.asa(::Type{DataFrame}, lv::LabeledValues)
    NeXLUncertainties.asa( #
       ::Type{DataFrame},
       uvss::UncertainValues,
       withCovars = true,
    )::DataFrame
"""
NeXLUncertainties.asa(::Type{DataFrame}, lv::LabeledValues) = DataFrame( Label = labels(lv), Value = values(lv))
"""
"""
function NeXLUncertainties.asa( #
    ::Type{DataFrame},
    uvss::UncertainValues,
    withCovars = true,
)::DataFrame
    lbls = labels(uvss)
    df = DataFrame(Variable = map(lbl -> "$lbl", lbls), Values = map(lbl -> value(uvss, lbl), lbls))
    if withCovars
        for (i, cl) in enumerate(lbls)
            insertcols!(
                df,
                2 + i,
                Symbol("$cl") => map(rl -> covariance(uvss, rl, cl), lbls),
            )
        end
    else
        insertcols!(df, 3, :σ => map(lbl -> σ(uvss, lbl), lbls))
    end
    return df
end

end # module