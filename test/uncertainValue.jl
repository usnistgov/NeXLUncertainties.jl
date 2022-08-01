using Test
using NeXLUncertainties
using LaTeXStrings

@testset "Uncertainty" begin
    uv1, uv2, uv3 =
        UncertainValue(1.0, 0.1), UncertainValue(0.8, 0.001), UncertainValue(-0.9, 0.05)

    @testset "Sigma" begin
        @test σ(uv1) == uncertainty(uv1)
        @test σ(1.0) == 0.0
        @test isequal(uv(1.0, 0.1), uv1)
        @test uv(1.0, 0.1) == uv1
        @test σ(uv1) == 0.1
        @test σ(uv2) == 0.001
        @test σ(1.0) == 0.0
        @test variance(uv2) == 0.001^2
        @test variance(1.0) == 0.0
        @test fractional(uv2) == 0.001 / 0.8
        @test fractional(uv3) == abs(0.05 / -0.9)
        @test fractional(0.0) == 0.0
        @test value(uv2) == 0.8
        @test value(uv3) == -0.9
        @test value(1.0) == 1.0

        @test min(uv1, uv2) === uv2
        @test min(uv2, uv3) === uv3
        @test min(uv1, uv3) === uv3
        @test min(uv1, UncertainValue(1.0, 0.05)) === uv1
        @test minimum([uv1, uv2, uv3]) === uv3
        @test minimum((uv1, uv2, uv3)) === uv3

        @test max(uv1, uv2) === uv1
        @test max(uv1, uv2) === uv1
        @test max(uv1, uv3) === uv1
        @test max(uv2, UncertainValue(0.8, 0.0005)) == uv2
        @test maximum([uv1, uv2, uv3]) === uv1
        @test maximum((uv1, uv2, uv3)) === uv1

        @test isless(uv1, uv2) == false
        @test isless(uv2, uv1) == true
        @test isless(uv(1.0, 0.1), uv(1.0, 0.2)) == false
        @test isless(uv(1.0, 0.2), uv(1.0, 0.1)) == true
    end

    @testset "Conversions" begin
        @test convert(UncertainValue, 2.0) == uv(2.0, 0.0)
        @test convert(UncertainValue, 2) == uv(2.0, 0.0)
        @test zero(UncertainValue) == uv(0.0, 0.0)
        @test one(UncertainValue) == uv(1.0, 0.0)

        @test -one(UncertainValue) == uv(-1.0, 0.0)
        @test +one(UncertainValue) == uv(1.0, 0.0)
    end

    @testset "Operations" begin
        # Values are taken from the NIST Uncertainty Machine (Gauss's Formula (GUM's Linear Approximation))
        @test isapprox(3.0 * uv2, uv(2.4, 0.003), atol = 1.0e-10)
        @test isapprox(uv2 * 3.0, uv(2.4, 0.003), atol = 1.0e-10)

        @test isapprox(inv(uv(5.0, 0.2)), uv(0.2, 0.008), atol = 1.0e-5)
        @test isapprox(1.0 / uv(5.0, 0.2), uv(0.2, 0.008), atol = 1.0e-5)

        @test isapprox(uv(2.0, 0.2)^5.0, uv(32.0, 16.0), atol = 1.0e-5)
        @test isapprox(log(uv(2.0, 0.2)), uv(0.693, 0.100), atol = 1.0e-3)
        @test isapprox(exp(uv(2.0, 0.2)), uv(7.39, 1.48), atol = 1.0e-2)
        @test isapprox(sin(uv(2.0, 0.2)), uv(0.909, 0.0832), atol = 1.0e-3)
        @test isapprox(cos(uv(2.0, 0.2)), uv(-0.416, 0.182), atol = 1.0e-3)
        @test isapprox(tan(uv(0.3, 0.1)), uv(0.309, 0.110), atol = 1.0e-3)
        @test isapprox(sqrt(uv(3.5, 0.1)), uv(1.871, 0.0267), atol = 1.0e-3)

        @test equivalent(uv(3.3, 0.2), uv(3.5, 0.1))
        @test !equivalent(uv(3.3, 0.2), uv(3.6, 0.1))

        @test isapprox(
            multiply(uv(2.3, 0.4), uv(3.1, 0.2), 0.3),
            uv(7.13, 1.45),
            atol = 0.01,
        )
        @test isapprox(
            divide(uv(2.3, 0.4), uv(3.1, 0.2), 0.3),
            uv(0.742, 0.123),
            atol = 0.001,
        )

        @test isapprox(
            multiply(uv(2.3, 0.4), uv(3.1, 0.2), -0.2),
            uv(7.13, 1.23),
            atol = 0.01,
        )

        @test isapprox(
            multiply(uv(2.3, 0.4), uv(2.3, 0.4), -1.0),
            uv(2.3*2.3, 0.0),
            atol = 0.001
        )

        @test isapprox(
            multiply(uv(2.3, 0.4), uv(2.3, 0.4), 1.0),
            uv(2.3*2.3, 1.84),
            atol = 0.001
        )

        @test isapprox(
            multiply(uv(2.3, 0.4), uv(2.3, 0.4), 0.0),
            uv(2.3*2.3, 1.301),
            atol = 0.001
        )

        @test isapprox(
            divide(uv(2.3, 0.4), uv(3.1, 0.2), -0.2),
            uv(0.74194, 0.146),
            atol = 0.001,
        )

        @test isapprox(
            divide(uv(2.3, 0.4), uv(2.3, 0.4), 1.0),
            uv(1.0, 0.0),
            atol = 0.001,
        )

        @test isapprox(
            divide(uv(2.3, 0.4), uv(2.3, 0.4), -1.0),
            uv(1.0, 0.348),
            atol = 0.001,
        )

        @test isapprox(
            divide(uv(2.3, 0.4), uv(2.3, 0.4), 0.0),
            uv(1.0, 0.245),
            atol = 0.001,
        )

        @test isapprox(
            add(-3.0, uv(2.3, 0.4), 4.0, uv(3.1, 0.2), -0.2),
            uv(5.5, 1.57),
            atol = 0.01,
        )
        @test isapprox(
            add(3.0, uv(2.3, 0.4), -4.0, uv(3.1, 0.2), 0.3),
            uv(-5.5, 1.23),
            atol = 0.01,
        )

        @test isequal(abs(uv(1.0, 0.2)), uv(1.0, 0.2))
        @test isequal(abs(uv(-1.0, 0.2)), uv(1.0, 0.2))
    end
    @testset "Parse" begin
        @test isequal(parse(UncertainValue, "1.0±0.1"), uv(1.0, 0.1))
        @test isequal(parse(UncertainValue, "2.0 ± 0.3"), uv(2.0, 0.3))
        @test isequal(parse(UncertainValue, "-1.0 ± 0.2"), uv(-1.0, 0.2))
        @test isequal(parse(UncertainValue, "12.3 +- 0.12"), uv(12.3, 0.12))
        @test isequal(parse(UncertainValue, "12.3 -+ 0.12"), uv(12.3, 0.12))
        @test isequal(parse(UncertainValue, repr(uv1)), uv1)
        @test isequal(parse(UncertainValue, repr(uv2)), uv2)
    end
    @testset "LaTeX" begin
        @test isequal(latexstring(uv(1.0,0.033),mode=:normal), L"$1.000 \pm 0.033$")
        @test isequal(latexstring(uv(1.0,0.033),mode=:normal), latexstring(uv(1.0,0.033)))
        @test isequal(latexstring(uv(1.0,0.033),mode=:siunitx), L"$\num{1.000 \pm 0.033}$")
        @test isequal(latexstring(uv(1.0,0.033),fmt = "%0.5f",mode=:normal), L"$1.00000 \pm 0.03300$")
        @test isequal(latexstring(uv(1.0,0.033),fmt = "%0.4f",mode = :siunitx), L"$\num{1.0000 \pm 0.0330}$")
    end
end
