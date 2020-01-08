using Test
using NeXLUncertainties
using Random

@testset "Model 1" begin

    struct TestMeasurementModel <: MeasurementModel end

    d(a, b, c) = a + b^2 + c^3
    e(a, b, c) = log(a) + exp(c)
    f(a, b, c) = 3.2 * a * b * c
    g(a, b, c) = 2.3 * a * b + 1.8 * a * c

    function NeXLUncertainties.compute(mm::TestMeasurementModel, inputs::LabeledValues, withJac::Bool)
        la, lb, lc = label.([ "A", "B", "C"])
        a, b, c = inputs[la], inputs[lb], inputs[lc]

        outputs = label.([ "D", "E", "F", "G" ])
        results = [d(a, b, c), e(a, b, c), f(a, b, c), g(a, b, c)]
        if withJac # Compute the Jacobian column-by-column (input-by-input)
            jac = zeros(length(outputs), length(inputs))
            jac[:, indexin(la, inputs)] = [1.0, 1.0 / a, results[3] / a, 2.3 * b + 1.8 * c]
            jac[:, indexin(lb, inputs)] = [2.0 * b, 0, results[3] / b, 2.3 * a]
            jac[:, indexin(lc, inputs)] = [3.0 * c^2, exp(c), results[3] / c, 1.8 * a]
        else
            jac = missing
        end
        return (LabeledValues(outputs, results), jac)
    end

    labels = Label[label("A"), label("B"), label("C")]
    a, b, c = 2.0, π / 8, -1.0  # values
    da, db, dc = 0.1, π / 40, 0.05 # uncertainties
    cab, cac, cbc = -0.3, 0.8, 0.1 # correlation coefficients
    values = [a, b, c]
    covars = [
        da^2 cab * da * db cac * da * dc
        cab * da * db db^2 cbc * db * dc
        cac * da * dc cbc * db * dc dc^2
    ]

    resc = compute(TestMeasurementModel(), LabeledValues(labels, values))

    @test resc[label("D")] == d(a, b, c)
    @test resc[label("E")] == e(a, b, c)
    @test resc[label("F")] == f(a, b, c)
    @test resc[label("G")] == g(a, b, c)

    inputs = uvs(labels, values, covars)
    res = propagate(TestMeasurementModel(), inputs)
    ld, le, lf, lg = label.(("D", "E", "F", "G"))

    @test value(res[ld]) == resc[ld]
    @test value(res[le]) == resc[le]
    @test value(res[lf]) == resc[lf]
    @test value(res[lg]) == resc[lg]

    println(res)
    println("D=$(d(a,b,c)) E=$(e(a,b,c)) F=$(f(a,b,c)) G=$(g(a,b,c))")

    mcres = NeXLUncertainties.mcpropagate(TestMeasurementModel(), inputs, 100000, rng = MersenneTwister(0xFEED))
    println(mcres)
    # Check if the analytical and Monte Carlo agree?
    @test isapprox(value(mcres[ld]), value(res[ld]), atol = 0.05 * σ(res[ld]))
    @test isapprox(value(mcres[le]), value(res[le]), atol = 0.05 * σ(res[le]))
    @test isapprox(value(mcres[lf]), value(res[lf]), atol = 0.05 * σ(res[lf]))
    @test isapprox(value(mcres[lg]), value(res[lg]), atol = 0.05 * σ(res[lg]))

    @test isapprox(covariance(ld, le, mcres), covariance(ld, le, res), atol = 0.05 * σ(res[ld]) * σ(res[le]))
    @test isapprox(covariance(ld, lf, mcres), covariance(ld, lf, res), atol = 0.05 * σ(res[ld]) * σ(res[lf]))
    @test isapprox(covariance(ld, lg, mcres), covariance(ld, lg, res), atol = 0.05 * σ(res[ld]) * σ(res[lg]))
    @test isapprox(covariance(le, lf, mcres), covariance(le, lf, res), atol = 0.05 * σ(res[le]) * σ(res[lf]))
    @test isapprox(covariance(le, lg, mcres), covariance(le, lg, res), atol = 0.05 * σ(res[le]) * σ(res[lg]))
    @test isapprox(covariance(lf, lg, mcres), covariance(lf, lg, res), atol = 0.05 * σ(res[lf]) * σ(res[lg]))
end;

# K-ratio test is a simple univariate model but with many inputs and multiple steps.
@testset "K-Ratio test" begin
    inputs = uvs(Dict(
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
        nl"I[low,unk]" => poisson(30),
        nl"I[high,unk]" => poisson(25),
        nl"I[peak,unk]" => poisson(5000),
        nl"t[low,unk]" => uv(3.0, 0.01),
        nl"t[high,unk]" => uv(6.0, 0.01),
        nl"t[peak,unk]" => uv(2.6, 0.01),
        nl"i[low,unk]" => uv(10.0, 0.1),
        nl"i[high,unk]" => uv(10.1, 0.1),
        nl"i[peak,unk]" => uv(9.9, 0.1),
        nl"R[low,unk]" => uv(102.0, 0.01),
        nl"R[high,unk]" => uv(108.0, 0.01),
        nl"R[peak,unk]" => uv(106.0, 0.01),
    ))

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
            jac[1, indexin(lI, inputs)] = results[1] / I
            jac[1, indexin(lt, inputs)] = -results[1] / t
            jac[1, indexin(li, inputs)] = -results[1] / i
        end
        return (LabeledValues(labels, results), jac)
    end

    struct IChar <: MeasurementModel
        sample::String
    end

    function NeXLUncertainties.compute(ic::IChar, inputs::LabeledValues, withJac::Bool)
        lNIl, lNIp, lNIh = label.(("NI[low,$(ic.sample)]", "NI[peak,$(ic.sample)]", "NI[high,$(ic.sample)]"))
        lRl, lRp, lRh = label.(("R[low,$(ic.sample)]", "R[peak,$(ic.sample)]", "R[high,$(ic.sample)]"))
        NIl, NIp, NIh = inputs[lNIl], inputs[lNIp], inputs[lNIh]
        Rl, Rp, Rh = inputs[lRl], inputs[lRp], inputs[lRh]
        labels = [label("Ichar[$(ic.sample)]")]
        results = [NIp - ((Rp - Rl) * NIh + (Rh - Rp) * NIl) / (Rh - Rl)]
        jac = withJac ? zeros(Float64, 1, length(inputs)) : missing
        if withJac
            jac[1, indexin(lNIp, inputs)] = 1.0
            jac[1, indexin(lNIl, inputs)] = (Rh - Rp) / (Rh - Rl)
            jac[1, indexin(lNIh, inputs)] = (Rl - Rp) / (Rh - Rl)
            jac[1, indexin(lRp, inputs)] = (NIl - NIh) / (Rh - Rl)
            jac[1, indexin(lRl, inputs)] = (NIh - NIl) * (Rh - Rp) / ((Rh - Rl)^2)
            jac[1, indexin(lRh, inputs)] = (NIl - NIh) * (Rl - Rp) / ((Rh - Rl)^2)
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
            jac[1, indexin(lIunk, inputs)] = results[1] / inputs[lIunk]
            jac[1, indexin(lIstd, inputs)] = -results[1] / inputs[lIstd]
        end
        return (LabeledValues(labels, results), jac)
    end

    retain = missing # AllInputs()
    norm2char(id) = MaintainInputs([label("R[$i,$id]") for i in ("low", "peak", "high")])
    ICharModel(id::String) = IChar(id) ∘ (NormI("low", id) | NormI("peak", id) | NormI("high", id) | norm2char(id))
    TotalModel(id::String) = (KRatioModel(id) | retain) ∘ (ICharModel("std") | ICharModel("unk") | retain)

    res = TotalModel("K-L3")(inputs)
    resmc = mcpropagate(TotalModel("K-L3"), inputs, 1000)

    lbl = nl"k[K-L3]"
    println("res   = ", value(res[lbl]), " ± ", σ(res[lbl]))
    println("resmc = ", value(resmc[lbl]), " ± ", σ(resmc[lbl]))

    @test isapprox(value(res[lbl]), 10.8994931770182, atol=1.0e-12) # As calculated in a spreadsheet
    @test isapprox(value(res[lbl]), value(resmc[lbl]), atol = 0.1)
    @test isapprox(σ(res[lbl]), σ(resmc[lbl]), atol = 0.1)
end;
