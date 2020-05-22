using ITensors.NDTensors,
      Test
using LinearAlgebra

@testset "random_orthog" begin
  n,m = 10,4
  O1 = random_orthog(n,m)
  @test norm(transpose(O1)*O1 - Diagonal(fill(1.,m))) < 1E-14

  n,m = 4,10
  O2 = random_orthog(n,m)
  @test norm(O2*transpose(O2) - Diagonal(fill(1.,n))) < 1E-14
end

@testset "random_unitary" begin
  n,m = 10,4
  U1 = random_unitary(n,m)
  @test eltype(U1) <: ComplexF64
  @test norm(U1'*U1 - Diagonal(fill(1.,m))) < 1E-14

  n,m = 4,10
  U2 = random_unitary(n,m)
  @test norm(U2*U2' - Diagonal(fill(1.,n))) < 1E-14
end

