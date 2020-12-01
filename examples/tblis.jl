using LinearAlgebra
using NDTensors
using TBLIS

let
  d = 25

  nthreads = 4

  A = randomTensor(d, d, d, d)
  B = randomTensor(d, d, d, d)
  C_blas = randomTensor(d, d, d, d)

  labelsA = (1, -1, 2, -2)
  labelsB = (-2, 4, 3, -1)
  labelsC = (1, 2, 3, 4)

  println("Contracting (d x d x d x d) tensors A * B -> C, d = ", d)
  @show labelsA
  @show labelsB
  @show labelsC

  #
  # Use BLAS
  #

  disable_tblis!()
  BLAS.set_num_threads(nthreads)

  println()
  println("Using BLAS with $nthreads threads")

  time_blas = @belapsed NDTensors.contract!($C_blas, $labelsC, $A, $labelsA, $B, $labelsB) samples = 100

  println()
  println("Time (BLAS) = ", time_blas, " seconds")

  #
  # Use TBLIS
  #

  enable_tblis!()
  TBLIS.set_num_threads(nthreads)

  println()
  println("Using TBLIS with $(TBLIS.get_num_threads()) threads")

  C_tblis = randomTensor(d, d, d, d)

  time_tblis = @belapsed NDTensors.contract!($C_tblis, $labelsC, $A, $labelsA, $B, $labelsB) samples = 100

  println()
  println("Time (TBLIS) = ", time_tblis, " seconds")

  println()
  @show C_blas ≈ C_tblis

  println()
  println("Time (TBLIS) / Time (BLAS) = ", time_tblis / time_blas)
end

