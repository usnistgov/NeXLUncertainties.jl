using Test
using NeXLUncertainties

@testset "UncertainValues" begin
    @testset "Label" begin
        l1a = label(123)
        l1b = label(123)
        l2a = nl"X"
        l2b = nl"X"
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

        cov = [
            0.1^2 -0.3*0.1*0.2 0.0
            -0.3*0.1*0.2 0.2^2 0.2*0.2*0.3
            0.0 0.2*0.2*0.3 0.3^2
        ]
        @test checkcovariance!(cov)

        bad = [
            0.1^2 -0.29*0.1*0.2 0.0
            -0.3*0.1*0.2 0.2^2 0.2*0.2*0.3
            0.0 0.2*0.2*0.3 0.3^2
        ]

        @test_throws ErrorException checkcovariance!(bad)

        lbls = [label("X0"), label("X1"), label("Z")]
        vals = [2.0, 8.0, 12.0]

        uvs1 = uvs(lbls, vals, cov)

        @test σ(uvs1, label("X0")) == 0.100
        @test σ(uvs1, label("X1")) == 0.200
        @test σ(uvs1, label("Z")) == 0.3
        @test σ(uvs1, label("Z")) == 0.3

        @test filter(uvs1, [label("Z"), label("X1")]) ==
              [cov[3, 3] cov[3, 2]; cov[2, 3] cov[2, 2]]

        @test uvs1[label("Z")] == uv(12.0, 0.3)
        @test uvs1[label("X1")] == uv(8.0, 0.2)
        @test get(uvs1, label("X1"), uv(8.1, 0.0)) == uv(8.0, 0.2)
        @test get(uvs1, label("Z"), uv(8.1, 0.0)) ≠ uv(8.0, 0.2)
        @test get(uvs1, label("Zz"), uv(8.1, 0.0)) == uv(8.1, 0.0)

        @test length(uvs1) == 3

        @test value(uvs1, label("Z")) == 12.0
        @test value(uvs1, label("X1")) == 8.0

        @test covariance(uvs1, label("X1"), label("Z")) == 0.2 * 0.2 * 0.3
        @test covariance(uvs1, label("Z"), label("X1")) == 0.2 * 0.2 * 0.3

        @test variance(uvs1, label("Z")) == 0.3^2
        @test variance(uvs1, label("X1")) == 0.2^2

        @test uncertainty(uvs1, label("Z")) == 0.3
        @test uncertainty(uvs1, label("X1")) == 0.2
        @test uncertainty(uvs1, label("Z"), 2.0) == 2.0 * 0.3
        @test uncertainty(uvs1, label("X1"), 3.0) == 3.0 * 0.2
    end

    @testset "cat" begin
        uvs1 = uvs(
            [nl"A", nl"B", nl"C"],
            [1.0, 2.0, 3.0],
            [0.1 0.05 -0.03; 0.05 0.2 0.06; -0.03 0.06 0.3],
        )
        uvs2 = uvs(
            [nl"D", nl"E", nl"F"],
            [4.0, 5.0, 6.0],
            [0.2 0.1 -0.13; 0.1 0.3 -0.12; -0.13 -0.12 0.1],
        )
        uvs3 = cat([uvs1, uvs2])
        @test value(uvs3, nl"A") == 1.0
        @test value(uvs3, nl"C") == 3.0
        @test value(uvs3, nl"A") == value(uvs1, nl"A")
        @test value(uvs3, nl"C") == value(uvs1, nl"C")
        @test value(uvs3, nl"E") == value(uvs2, nl"E")
        @test value(uvs3, nl"F") == value(uvs2, nl"F")
        @test covariance(uvs3, nl"A", nl"B") == 0.05
        @test covariance(uvs3, nl"E", nl"E") == variance(uvs3, nl"E")
        @test covariance(uvs3, nl"E", nl"E") == 0.3
        @test covariance(uvs3, nl"E", nl"F") == -0.12
        @test covariance(uvs3, nl"B", nl"D") == 0.0
        @test covariance(uvs3, nl"E", nl"E") == variance(uvs3, nl"E")
        @test σ(uvs3, nl"E") == sqrt(0.3)
        @test σ(uvs2, nl"E") == σ(uvs3, nl"E")

        ex = filter(uvs3, [nl"A", nl"C", nl"E"])
        @test ex[1, 1] == covariance(uvs3, nl"A", nl"A")
        @test ex[1, 2] == covariance(uvs3, nl"A", nl"C")
        @test ex[1, 2] == ex[2, 1]
        @test ex[3, 1] == covariance(uvs3, nl"E", nl"A")
        @test ex[1, 3] == ex[3, 1]

        @test uvs3[nl"E"] == UncertainValue(5.0, sqrt(0.3))
        @test uvs3[nl"C"] == UncertainValue(3.0, sqrt(0.3))
        @test get(uvs3, nl"C", missing) == UncertainValue(3.0, sqrt(0.3))
        @test ismissing(get(uvs3, nl"XX", missing))
        @test get(uvs3, nl"C", missing) == uvs3[nl"C"]
        @test get(uvs3, nl"C", UncertainValue(3.2, sqrt(0.99))) == uvs3[nl"C"]
        @test get(uvs3, nl"zZ", UncertainValue(3.2, sqrt(0.99))) ==
              UncertainValue(3.2, sqrt(0.99))

        @test length(uvs3) == 6
        @test length(uvs1) == 3

        vals = values(uvs3)
        lbls = labels(uvs3)
        @test value(uvs3[lbls[1]]) == vals[1]
        @test value(uvs3[lbls[3]]) == vals[3]

        @test uncertainty(uvs3, nl"A", 2.0) == 2.0 * sqrt(0.1)
    end
end