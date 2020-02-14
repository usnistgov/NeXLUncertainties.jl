


function NeXLUncertainties.compute(st::Step, inputs::LabeledValues, withJac::Bool)::MMResult
    # Build input variable labels
    xl, yl = xLabel(), yLabel()
    # Extract input variables
    x, y = inputs[xl], inputs[yl]
    # Compute the values
    a = fa(x,y)
    b = fb(x,y)
    vals = LabeledValues([ aLabel(), bLabel() ],[ a, b ])
    jac = withJac ? zeros(Float64, length(vals), length(inputs)) : missing
    if withJac
        jac[indexin(aLabel(),vals), indexin(xl,inputs)] = δfaδx(x,y)
        jac[indexin(aLabel(),vals), indexin(yl,inputs)] = δfaδy(x,y)
        jac[indexin(bLabel(),vals), indexin(xl,inputs)] = δfbδx(x,y)
        jac[indexin(bLabel(),vals), indexin(yl,inputs)] = δfbδy(x,y)
    end
    return (vals, jac)
end
