using NeXLUncertainties
using BenchmarkTools

inputs = uvs(
    Dict(
        nl"I[low,std]" => poisson(20),
        nl"I[high,std]" => poisson(15),
        nl"I[peak,std]" => poisson(1000),
        nl"t[low,std]" => uv(5.0, 0.01),
        nl"t[high,std]" => uv(8.0, 0.01),
        nl"t[peak,std]" => uv(5.6, 0.01),
        nl"i[low,std]" => uv(10.0, 0.1),
        nl"i[high,std]" => uv(10.1, 0.1),
        nl"i[peak,std]" => uv(9.9, 0.1),
        nl"R[low,std]" => uv(100.0, 0.01),
        nl"R[high,std]" => uv(110.0, 0.01),
        nl"R[peak,std]" => uv(106.0, 0.01),
        nl"I[low,unk]" => uv(30.0, sqrt(30.0)),
        nl"I[high,unk]" => uv(25.0, sqrt(25.0)),
        nl"I[peak,unk]" => uv(5000.0, sqrt(5000.0)),
        nl"t[low,unk]" => uv(3.0, 0.01),
        nl"t[high,unk]" => uv(6.0, 0.01),
        nl"t[peak,unk]" => uv(2.6, 0.01),
        nl"i[low,unk]" => uv(10.0, 0.1),
        nl"i[high,unk]" => uv(10.1, 0.1),
        nl"i[peak,unk]" => uv(9.9, 0.1),
        nl"R[low,unk]" => uv(102.0, 0.01),
        nl"R[high,unk]" => uv(108.0, 0.01),
        nl"R[peak,unk]" => uv(106.0, 0.01),
    ),
)

struct NormI <: MeasurementModel
    position::String
    sample::String
end

function NeXLUncertainties.compute(ni::NormI, inputs::LabeledValues, withJac::Bool)
    lI, lt, li = label("I[$(ni.position),$(ni.sample)]"),
    label("t[$(ni.position),$(ni.sample)]"),
    label("i[$(ni.position),$(ni.sample)]")
    I, t, i = inputs[lI], inputs[lt], inputs[li]
    labels = [label("NI[$(ni.position),$(ni.sample)]")]
    results = [I / (t * i)]
    jac = withJac ? zeros(Float64, 1, length(inputs)) : missing
    if withJac
        jac[1, indexin(inputs, lI)] = results[1] / I
        jac[1, indexin(inputs, lt)] = -results[1] / t
        jac[1, indexin(inputs, li)] = -results[1] / i
    end
    return (LabeledValues(labels, results), jac)
end

struct IChar <: MeasurementModel
    sample::String
end

function NeXLUncertainties.compute(ic::IChar, inputs::LabeledValues, withJac::Bool)
    lNIl, lNIp, lNIh =
        label.(("NI[low,$(ic.sample)]", "NI[peak,$(ic.sample)]", "NI[high,$(ic.sample)]"))
    lRl, lRp, lRh =
        label.(("R[low,$(ic.sample)]", "R[peak,$(ic.sample)]", "R[high,$(ic.sample)]"))
    NIl, NIp, NIh = inputs[lNIl], inputs[lNIp], inputs[lNIh]
    Rl, Rp, Rh = inputs[lRl], inputs[lRp], inputs[lRh]
    labels = [label("Ichar[$(ic.sample)]")]
    results = [NIp - ((Rp - Rl) * NIh + (Rh - Rp) * NIl) / (Rh - Rl)]
    jac = withJac ? zeros(Float64, 1, length(inputs)) : missing
    if withJac
        jac[1, indexin(inputs, lNIp)] = 1.0
        jac[1, indexin(inputs, lNIl)] = (Rh - Rp) / (Rh - Rl)
        jac[1, indexin(inputs, lNIh)] = (Rl - Rp) / (Rh - Rl)
        jac[1, indexin(inputs, lRp)] = (NIl - NIh) / (Rh - Rl)
        jac[1, indexin(inputs, lRl)] = (NIh - NIl) * (Rh - Rp) / ((Rh - Rl)^2)
        jac[1, indexin(inputs, lRh)] = (NIl - NIh) * (Rl - Rp) / ((Rh - Rl)^2)
    end
    return (LabeledValues(labels, results), jac)
end

struct KRatioModel <: MeasurementModel
    id::String
end

function NeXLUncertainties.compute(kr::KRatioModel, inputs::LabeledValues, withJac::Bool)
    lIstd, lIunk = nl"Ichar[std]", nl"Ichar[unk]"
    labels = [label("k[$(kr.id)]")]
    results = [inputs[lIunk] / inputs[lIstd]]
    jac = withJac ? zeros(Float64, 1, length(inputs)) : missing
    if withJac
        jac[1, indexin(inputs, lIunk)] = results[1] / inputs[lIunk]
        jac[1, indexin(inputs, lIstd)] = -results[1] / inputs[lIstd]
    end
    return (LabeledValues(labels, results), jac)
end

retain = missing # AllInputs()
norm2char(id) = MaintainInputs([label("R[$i,$id]") for i in ("low", "peak", "high")])
ICharModel(id::String) =
    IChar(id) ∘ (NormI("low", id) | NormI("peak", id) | NormI("high", id) | norm2char(id))
TotalModel(id::String) =
    (KRatioModel(id) | retain) ∘ (ICharModel("std") | ICharModel("unk") | retain)

res = TotalModel("K-L3")(inputs)
resmc = mcpropagate(TotalModel("K-L3"), inputs, 1000)

@btime mcpropagate(TotalModel("K-L3"), inputs, 1000)
@btime TotalModel("K-L3")(inputs)

lbl = nl"k[K-L3]"
println("res   = ", value(res[lbl]), " ± ", σ(res[lbl]))
println("resmc = ", value(resmc[lbl]), " ± ", σ(resmc[lbl]))
