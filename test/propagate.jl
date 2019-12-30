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
        return (outputs, results, jac)
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

    mcres = NeXLUncertainties.mcpropagate(TestMeasurementModel(), inputs, 100000, MersenneTwister(0xFEED))
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
