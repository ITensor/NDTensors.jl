
#
# Empty storage
#

struct Empty{ElT, StoreT <: TensorStorage} <: TensorStorage{ElT}
end

# Get the Empty version of the TensorStorage
function empty(::Type{StoreT}) where {StoreT <: TensorStorage{ElT}} where {ElT}
  return Empty{ElT, StoreT}
end

# Defaults to Dense
function Empty(::Type{ElT}) where {ElT}
  return empty(Dense{ElT, Vector{ElT}})()
end

Empty() = Empty(Float64)

copy(S::Empty) = S

isempty(::Empty) = true

nnzblocks(::Empty) = 0

nnz(::Empty) = 0

function complex(::Type{<: Empty{ElT, StoreT}}) where {ElT,
                                                            StoreT}
  return Empty{complex(ElT), complex(StoreT)}
end

function complex(S::Empty)
  return complex(typeof(S))()
end

#size(::Empty) = 0

function show(io::IO,
                   mime::MIME"text/plain",
                   S::Empty)
  println(io, typeof(S))
end

#
# EmptyTensor (Tensor using Empty storage)
#

const EmptyTensor{ElT, N, StoreT, IndsT} =
  Tensor{ElT, N, StoreT, IndsT} where {StoreT <: Empty}

isempty(::EmptyTensor) = true

EmptyTensor(::Type{ElT}, inds) where {ElT <: Number} =
  tensor(Empty(ElT), inds)

EmptyTensor(inds) = EmptyTensor(Float64, inds)

EmptyTensor(inds::Int...) = EmptyTensor(inds)

function EmptyTensor(::Type{StoreT}, inds) where {StoreT <: TensorStorage}
  return tensor(empty(StoreT)(), inds)
end

function EmptyBlockSparseTensor(::Type{ElT}, inds) where {ElT <: Number}
  StoreT = BlockSparse{ElT, Vector{ElT}, length(inds)}
  return EmptyTensor(StoreT, inds)
end

# From an EmptyTensor, return the closest Tensor type
function fill(::Type{<: Tensor{ElT, N, EStoreT, IndsT}}) where {ElT <: Number, N, EStoreT <: Empty{ElT, StoreT}, IndsT} where {StoreT}
  return Tensor{ElT, N, StoreT, IndsT}
end

function zeros(T::TensorT) where {TensorT <: EmptyTensor}
  TensorR = fill(TensorT)
  return zeros(TensorR, inds(T))
end

function insertblock(T::EmptyTensor{<: Number, N}, block) where {N}
  R = zeros(T)
  insertblock!(R, Block(block))
  return R
end

@propagate_inbounds function setindex(T::EmptyTensor{<: Number, N},
                                      x, I...) where {N}
  R = zeros(T)
  R[I...] = x
  return R
end

function show(io::IO,
                   mime::MIME"text/plain",
                   T::EmptyTensor)
  summary(io, T)
  println(io)
end

