module NDTensors

using Compat
using Random
using LinearAlgebra
using StaticArrays
using HDF5
using Requires
using Strided

#####################################
# Exports
#
include("exports.jl")

#####################################
# DenseTensor and DiagTensor
#
include("tupletools.jl")
include("dims.jl")
include("tensorstorage.jl")
include("tensor.jl")
include("contraction_logic.jl")
include("dense.jl")
include("symmetric.jl")
include("linearalgebra.jl")
include("diag.jl")
include("combiner.jl")
include("truncate.jl")
include("svd.jl")

#####################################
# BlockSparseTensor
#
include("blocksparse/blockdims.jl")
include("blocksparse/blockoffsets.jl")
include("blocksparse/blocksparse.jl")
include("blocksparse/blocksparsetensor.jl")
include("blocksparse/diagblocksparse.jl")
include("blocksparse/combiner.jl")
include("blocksparse/linearalgebra.jl")

#####################################
# Empty
#
include("empty.jl")

#####################################
# Optional TBLIS contraction backend
#
function __init__()
  @require TBLIS="48530278-0828-4a49-9772-0f3830dfa1e9" include("tblis.jl")
end

end # module NDTensors
