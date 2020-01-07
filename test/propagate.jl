using Test
using NeXLUncertainties
using Random

@testset "Model 1" begin

    struct TestMeasurementModel <: MeasurementModel end

    d(a,b,c) = a + b^2 + c^3
    e(a,b,c) = log(a)+exp(c)
    f(a,b,c) = 3.2*a*b*c
    g(a,b,c) = 2.3*a*b + 1.8*a*c

    function NeXLUncertainties.compute(mm::TestMeasurementModel, inputs::LabeledValues, withJac::Bool)
        la, lb, lc = label("A"), label("B"), label("C")
        a, b, c =  inputs[la], inputs[lb], inputs[lc]

        outputs = [ label("D"), label("E"), label("F"), label("G") ]
        results =  [ d(a,b,c), e(a,b,c), f(a, b, c), g(a, b, c) ]
        if withJac # Compute the Jacobian column-by-column (input-by-input)
            jac = zeros(length(outputs), length(inputs))
            jac[:, indexin(la, inputs)] = [ 1.0, 1.0/a, results[3]/a, 2.3*b + 1.8*c]
            jac[:, indexin(lb, inputs)] = [ 2.0*b, 0, results[3]/b, 2.3*a ]
            jac[:, indexin(lc, inputs)] = [ 3.0*c^2, exp(c), results[3]/c, 1.8*a ]
        else
            jac=missing
        end
        return (LabeledValues(outputs, results), jac)
    end

    labels = Label[ label("A"), label("B"), label("C") ]
    a, b, c = 2.0, π/8, -1.0  # values
    da, db, dc = 0.1, π/40, 0.05 # uncertainties
    cab, cac, cbc = -0.3, 0.8, 0.1 # correlation coefficients
    values = [ a, b, c ]
    covars = [     da^2    cab*da*db  cac*da*dc ;
               cab*da*db     db^2     cbc*db*dc ;
               cac*da*dc  cbc*db*dc    dc^2 ]

    resc = compute(TestMeasurementModel(), LabeledValues(labels, values))

    @test resc[label("D")] == d(a,b,c)
    @test resc[label("E")] == e(a,b,c)
    @test resc[label("F")] == f(a,b,c)
    @test resc[label("G")] == g(a,b,c)

    inputs = uvs(labels, values, covars)
    res = propagate(TestMeasurementModel(), inputs)
    ld, le, lf, lg = label.( ( "D", "E", "F", "G" ))

    @test value(res[ld]) == resc[ld]
    @test value(res[le]) == resc[le]
    @test value(res[lf]) == resc[lf]
    @test value(res[lg]) == resc[lg]

    println(res)
    println("D=$(d(a,b,c)) E=$(e(a,b,c)) F=$(f(a,b,c)) G=$(g(a,b,c))")

    mcres = NeXLUncertainties.mcpropagate(TestMeasurementModel(), inputs, 100000, rng=MersenneTwister(0xFEED))
    println(mcres)
    # Check if the analytical and Monte Carlo agree?
    @test isapprox(value(mcres[ld]), value(res[ld]), atol = 0.05*σ(res[ld]))
    @test isapprox(value(mcres[le]), value(res[le]), atol = 0.05*σ(res[le]))
    @test isapprox(value(mcres[lf]), value(res[lf]), atol = 0.05*σ(res[lf]))
    @test isapprox(value(mcres[lg]), value(res[lg]), atol = 0.05*σ(res[lg]))

    @test isapprox(covariance(ld,le,mcres),covariance(ld,le,res),atol=0.05*σ(res[ld])*σ(res[le]))
    @test isapprox(covariance(ld,lf,mcres),covariance(ld,lf,res),atol=0.05*σ(res[ld])*σ(res[lf]))
    @test isapprox(covariance(ld,lg,mcres),covariance(ld,lg,res),atol=0.05*σ(res[ld])*σ(res[lg]))
    @test isapprox(covariance(le,lf,mcres),covariance(le,lf,res),atol=0.05*σ(res[le])*σ(res[lf]))
    @test isapprox(covariance(le,lg,mcres),covariance(le,lg,res),atol=0.05*σ(res[le])*σ(res[lg]))
    @test isapprox(covariance(lf,lg,mcres),covariance(lf,lg,res),atol=0.05*σ(res[lf])*σ(res[lg]))
end;

@testset "K-Ratio test" begin
    inputs = uvs(Dict(
        nl"I[low,std]"=>uv(20.0,sqrt(20.0)),
        nl"I[high,std]"=>uv(15.0,sqrt(15.0)),
        nl"I[peak,std]"=>uv(1000.0,sqrt(1000.0)),
        nl"t[low,std]"=>uv(5.0,0.01),
        nl"t[high,std]"=>uv(8.0,0.01),
        nl"t[peak,std]"=>uv(5.6,0.01),
        nl"i[low,std]"=>uv(10.0,0.1),
        nl"i[high,std]"=>uv(10.1,0.1),
        nl"i[peak,std]"=>uv(9.9,0.1),
        nl"R[low,std]"=>uv(100.0,0.01),
        nl"R[high,std]"=>uv(110.0,0.01),
        nl"R[peak,std]"=>uv(106.0,0.01),
        nl"I[low,unk]"=>uv(30.0,sqrt(30.0)),
        nl"I[high,unk]"=>uv(25.0,sqrt(25.0)),
        nl"I[peak,unk]"=>uv(5000.0,sqrt(5000.0)),
        nl"t[low,unk]"=>uv(3.0,0.01),
        nl"t[high,unk]"=>uv(6.0,0.01),
        nl"t[peak,unk]"=>uv(2.6,0.01),
        nl"i[low,unk]"=>uv(10.0,0.1),
        nl"i[high,unk]"=>uv(10.1,0.1),
        nl"i[peak,unk]"=>uv(9.9,0.1),
        nl"R[low,unk]"=>uv(102.0,0.01),
        nl"R[high,unk]"=>uv(108.0,0.01),
        nl"R[peak,unk]"=>uv(106.0,0.01),
    ))

    struct NormI <: MeasurementModel
        position::String
        sample::String
    end

    function NeXLUncertainties.compute(ni::NormI, inputs::LabeledValues, withJac::Bool)
        lI, lt, li = label("I[$(ni.position),$(ni.sample)]"), label("t[$(ni.position),$(ni.sample)]"), label("i[$(ni.position),$(ni.sample)]")
        I, t, i = inputs[lI], inputs[lt], inputs[li]
        labels = [ label("NI[$(ni.position),$(ni.sample)]") ]
        results = [ I/(t*i) ]
        jac = withJac ? zeros(Float64, 1, length(inputs)) : missing
        if withJac
            jac[1, indexin(lI, inputs)] = 1.0/(t*i)
            jac[1, indexin(lt, inputs)] = -I*i/((t*i)^2)
            jac[1, indexin(li, inputs)] = -I*t/((t*i)^2)
        end
        return ( LabeledValues(labels, results), jac )
    end

    struct IChar <: MeasurementModel
        sample::String
    end

    function NeXLUncertainties.compute(ic::IChar, inputs::LabeledValues, withJac::Bool)
        lNIl, lNIp, lNIh = label.( [ "NI[low,$(ic.sample)]", "NI[peak,$(ic.sample)]", "NI[high,$(ic.sample)]" ] )
        lRl, lRp, lRh = label.( [ "R[low,$(ic.sample)]", "R[peak,$(ic.sample)]", "R[high,$(ic.sample)]" ] )
        NIl, NIp, NIh = inputs[lNIl], inputs[lNIp], inputs[lNIh]
        Rl, Rp, Rh = inputs[lRl], inputs[lRp], inputs[lRh]
        labels= [ label("Ichar[$(ic.sample)]") ]
        results = [ NIp - (Rp-Rl)*(NIp-NIl)/(Rh-Rl) ]
        jac = withJac ? zeros(Float64, 1, length(inputs)) : missing
        if withJac
            jac[1, indexin(lNIl, inputs)] = (Rp-Rl)/(Rh-Rl)
            jac[1, indexin(lNIp, inputs)] = 1.0
            jac[1, indexin(lNIh, inputs)] = -(Rp-Rl)/(Rh-Rl)
            jac[1, indexin(lRl, inputs)] = (NIp-NIl)*(1.0/(Rh-Rl) + (Rp-Rl)/((Rh-Rl)^2))
            jac[1, indexin(lRp, inputs)] = 1.0/(Rh-Rl)
            jac[1, indexin(lRh, inputs)] = -(Rp-Rl)/((Rh-Rl)^2)
        end
        return ( LabeledValues(labels, results), jac)
    end

    struct KRatioModel <: MeasurementModel
        id::String
    end

    function NeXLUncertainties.compute(kr::KRatioModel, inputs::LabeledValues, withJac::Bool)
        lIstd, lIunk = label("Ichar[std]"), label("Ichar[unk]")
        labels = [ label("k[$(kr.id)]") ]
        results = [ inputs[lIunk]/inputs[lIstd] ]
        jac = withJac ? zeros(Float64, 1, length(inputs)) : missing
        if withJac
            jac[1, indexin(lIstd, inputs)] = -results[1] / inputs[lIstd]
            jac[1, indexin(lIunk, inputs)] = results[1] / inputs[lIunk]
        end
        return ( LabeledValues(labels,results), jac )
    end

    println(stderr,inputs)

    ICharModel(id::String) = IChar(id) ∘ ( NormI("low",id) | NormI("peak",id) | NormI("high",id))
    TotalModel(id::String) = KRatioModel(id) ∘ ( ICharModel("std") | ICharModel("unk") )
    (vals,jac) = TotalModel("K-L3")(inputs)

end
