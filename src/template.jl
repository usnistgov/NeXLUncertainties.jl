
struct Model <: MeasurementModel end
struct XLabel <: Label end
struct YLabel <: Label end
struct ALabel <: Label end
struct BLabel <: Label end


function NeXLUncertainties.compute(st::Model, inputs::LabeledValues, withJac::Bool)::MMResult
    # Build input variable labels
    xl, yl = XLabel(), YLabel()
    # Extract input variables
    x, y = inputs[xl], inputs[yl]
    # Compute the values
    a = fa(x,y)
    b = fb(x,y)
    al, bl = ALabel(), BLabel()
    vals = LabeledValues([ al, bl ],[ a, b ])
    jac = withJac ? zeros(Float64, length(vals), length(inputs)) : missing
    if withJac
        jac[indexin(al,vals), indexin(xl,inputs)] = δfaδx(x,y)
        jac[indexin(al,vals), indexin(yl,inputs)] = δfaδy(x,y)
        jac[indexin(bl,vals), indexin(xl,inputs)] = δfbδx(x,y)
        jac[indexin(bl,vals), indexin(yl,inputs)] = δfbδy(x,y)
    end
    return (vals, jac)
end
