using Test
using NeXLUncertainties

using SparseArrays

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
