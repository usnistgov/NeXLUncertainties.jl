using Test
using NeXLUncertainties

using SparseArrays

@testset "Uncertainty" begin

        uv1, uv2, uv3 = UncertainValue(1.0,0.1), UncertainValue(0.8,0.001), UncertainValue(-0.9,0.05)

        @testset "Sigma" begin
                @test σ(uv1)==uncertainty(uv1)
                @test isequal(uv(1.0,0.1),uv1)
                @test σ(uv1)==0.1
                @test σ(uv1)==0.1
                @test variance(uv2)==0.001^2
                @test fractional(uv2)==0.001/0.8
                @test fractional(uv3)==0.05/-0.9
                @test value(uv2)==0.8
                @test value(uv3)==-0.9
        end

        @testset "Conversions" begin
                @test convert(UncertainValue,2.0) == uv(2.0,0.0)
                @test convert(UncertainValue,2) == uv(2.0,0.0)
                @test zero(UncertainValue) == uv(0.0,0.0)
                @test one(UncertainValue) == uv(1.0,0.0)

                @test -one(UncertainValue)==uv(-1.0,0.0)
                @test +one(UncertainValue)==uv(1.0,0.0)

                @test bound(1.0, 10.0, 100.0)==10.0
                @test bound(20.0, 10.0, 100.0)==20.0
                @test bound(200.0, 10.0, 100.0)==100.0
                @test bound(1.0, -100.0, -10.0)==-10.0
                @test bound(-20.0, -100.0, -10.0)==-20.0
                @test bound(-200.0, -100.0, -10.0)==-100.0
        end

        @testset "Operations" begin
                # Values are taken from the NIST Uncertainty Machine (Gauss's Formula (GUM's Linear Approximation))
                @test isapprox(3.0*uv2, uv(2.4,0.003),atol=1.0e-10)
                @test isapprox(uv2*3.0, uv(2.4,0.003),atol=1.0e-10)

                @test isapprox(inv(uv(5.0,0.2)),uv(0.2,0.008),atol=1.0e-5)
                @test isapprox(1.0/uv(5.0,0.2),uv(0.2,0.008),atol=1.0e-5)

                @test isapprox(divide(uv(5.0,0.2),3.0),uv(5.0/3.0,0.2/3.0),atol=1.0e-10)

                @test isapprox(power(uv(2.0,0.2),5.0),uv(32.0,16.0),atol=1.0e-5)
                @test isapprox(uv(2.0,0.2)^5.0,uv(32.0,16.0),atol=1.0e-5)
                @test isapprox(log(uv(2.0,0.2)),uv(0.693,0.100),atol=1.0e-3)
                @test isapprox(exp(uv(2.0,0.2)),uv(7.39,1.48),atol=1.0e-2)
                @test isapprox(sin(uv(2.0,0.2)),uv(0.909,0.0832),atol=1.0e-3)
                @test isapprox(cos(uv(2.0,0.2)),uv(-0.416,0.182),atol=1.0e-3)
                @test isapprox(tan(uv(0.3,0.1)),uv(0.309,0.110),atol=1.0e-3)
                @test isapprox(sqrt(uv(3.5,0.1)),uv(1.871,0.0267),atol=1.0e-3)

                @test equivalent(uv(3.3,0.2),uv(3.5,0.1))
                @test !equivalent(uv(3.3,0.2),uv(3.6,0.1))
        end

        @testset "Label" begin
                l1a = label(123)
                l1b = label(123)
                l2a = label("X")
                l2b = label("X")
                l3a = label(123.0)
                l3b = label(123.0)
                l3c = label(123.1)
                @test isequal(l1a, l1b)
                @test isequal(l2a, l2b)
                @test isequal(l3a, l3b)
                @test !isequal(l1a, l2a)
                @test !isequal(l1a, l3a)
                @test !isequal(l2a, l3a)
                @test !isequal(l3a, l3c)
        end

        @testset "Covariances" begin

                cov = [ 0.1^2          -0.3*0.1*0.2   0.0;
                        -0.3*0.1*0.2   0.2^2          0.2*0.2*0.3;
                        0.0            0.2*0.2*0.3    0.3^2
                 ]
                @test checkcovariance!(cov)
                @test checkcovariance!(sparse(cov))

                bad = [ 0.1^2          -0.29*0.1*0.2  0.0;
                        -0.3*0.1*0.2   0.2^2          0.2*0.2*0.3;
                        0.0            0.2*0.2*0.3    0.3^2
                 ]

                @test_throws ErrorException checkcovariance!(bad)
                @test_throws ErrorException checkcovariance!(sparse(bad))

                lbls = [label("X0"),label("X1"),label("Z")]
                vals=[2.0, 8.0,  12.0]

                uvs1=uvs(lbls,vals,cov)

                @test σ(label("X0"), uvs1)==0.100
                @test σ(label("X1"), uvs1)==0.200
                @test σ(label("Z"), uvs1)==0.3
                @test σ(label("Z"), uvs1)==0.3

                @test extract(uvs1, [label("Z"), label("X1")])==[ cov[3,3] cov[3,2]; cov[2,3] cov[2,2] ]

                @test uvs1[label("Z")]==uv(12.0,0.3)
                @test uvs1[label("X1")]==uv(8.0,0.2)
                @test get(uvs1,label("X1"),uv(8.1,0.0))==uv(8.0,0.2)
                @test get(uvs1,label("Z"),uv(8.1,0.0)) ≠ uv(8.0,0.2)
                @test get(uvs1,label("Zz"),uv(8.1,0.0)) == uv(8.1,0.0)

                @test length(uvs1)==3
                @test size(uvs1) == (3,)

                @test value(label("Z"),uvs1)==12.0
                @test value(label("X1"),uvs1)==8.0

                @test covariance(label("X1"), label("Z"), uvs1)==0.2*0.2*0.3
                @test covariance(label("Z"), label("X1"), uvs1)==0.2*0.2*0.3

                @test variance(label("Z"), uvs1)==0.3^2
                @test variance(label("X1"), uvs1)==0.2^2

                @test uncertainty(label("Z"), uvs1)==0.3
                @test uncertainty(label("X1"), uvs1)==0.2
                @test uncertainty(label("Z"), uvs1, 2.0)==2.0*0.3
                @test uncertainty(label("X1"), uvs1, 3.0)==3.0*0.2
        end
end
