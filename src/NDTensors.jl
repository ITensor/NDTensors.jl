module NDTensors

using Compat
using Dictionaries
using Random
using LinearAlgebra
using StaticArrays
using HDF5
using Requires
using Strided
using TimerOutputs
using TupleTools

using Base:
  @propagate_inbounds,
  ReshapedArray

using Base.Cartesian:
  @nexprs

using Base.Threads:
  @threads,
  nthreads

#####################################
# Imports and exports
#
include("exports.jl")
include("imports.jl")

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
include("blocksparse/block.jl")
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
# Deprecations
#
include("deprecated.jl")

#####################################
# A global timer used with TimerOutputs.jl
#

const timer = TimerOutput()

#####################################
# Optional block sparse multithreading
#

const _using_threaded_blocksparse = Ref(false)

using_threaded_blocksparse() = _using_threaded_blocksparse[]

function enable_threaded_blocksparse()
  _using_threaded_blocksparse[] = true
  return nothing
end

function disable_threaded_blocksparse()
  _using_threaded_blocksparse[] = false
  return nothing
end

#####################################
# Optional TBLIS contraction backend
#

const _using_tblis = Ref(false)

using_tblis() = _using_tblis[]

function enable_tblis()
  _using_tblis[] = true
  return nothing
end

function disable_tblis()
  _using_tblis[] = false
  return nothing
end

function __init__()
  @require TBLIS="48530278-0828-4a49-9772-0f3830dfa1e9" begin
    enable_tblis()
    include("tblis.jl")
  end
end

end # module NDTensors
