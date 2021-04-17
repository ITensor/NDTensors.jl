
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

Base.real(::Type{<: Empty{ElT, StoreT}}) where {ElT,StoreT} =
  Empty{real(ElT),real(StoreT)}

Base.real(S::Empty) = real(typeof(S))()

complex(::Type{<: Empty{ElT, StoreT}}) where {ElT,StoreT} = 
  Empty{complex(ElT), complex(StoreT)}

complex(S::Empty) = complex(typeof(S))()

#size(::Empty) = 0

function show(io::IO,
                   mime::MIME"text/plain",
                   S::Empty)
  println(io, typeof(S))
end

#
# EmptyTensor (Tensor using Empty storage)
#

const EmptyTensor{ElT,
                  N,
                  StoreT,
                  IndsT} = Tensor{ElT,
                                  N,
                                  StoreT,
                                  IndsT} where {StoreT <: Empty}

isempty(::EmptyTensor) = true

function EmptyTensor(::Type{ElT}, inds) where {ElT <: Number}
  return tensor(Empty(ElT), inds)
end

function EmptyTensor(::Type{StoreT}, inds) where {StoreT <: TensorStorage}
  return tensor(empty(StoreT)(), inds)
end

function EmptyBlockSparseTensor(::Type{ElT}, inds) where {ElT <: Number}
  StoreT = BlockSparse{ElT, Vector{ElT}, length(inds)}
  return EmptyTensor(StoreT, inds)
end

# From an EmptyTensor, return the closest Tensor type
function fill(::Type{<: Tensor{ElT,
                                    N,
                                    EStoreT,
                                    IndsT}}) where {ElT <: Number,
                                                    N,
                                                    EStoreT <: Empty{ElT,
                                                                     StoreT},
                                                    IndsT} where {StoreT}
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

insertblock!!(T::EmptyTensor{<: Number, N}, block) where {N} =
  insertblock(T, block)

@propagate_inbounds function _setindex(T::EmptyTensor, x, I...)
  R = zeros(T)
  R[I...] = x
  return R
end

@propagate_inbounds function setindex(T::EmptyTensor, x, I...)
  return _setindex(T, x, I...)
end

# This is needed to fix an ambiguity error with ArrayInterface.jl
# https://github.com/ITensor/NDTensors.jl/issues/62
@propagate_inbounds function setindex(T::EmptyTensor, x, I::Int...)
  return _setindex(T, x, I...)
end

setindex!!(T::EmptyTensor, x, I...) = setindex(T, x, I...)

# Version of contraction where output storage is empty
function contract!!(R::EmptyTensor{<:Number, NR}, labelsR::NTuple{NR},
                    T1::Tensor{<:Number, N1}, labelsT1::NTuple{N1},
                    T2::Tensor{<:Number, N2}, labelsT2::NTuple{N2}) where {NR, N1, N2}
  RR = contract(T1, labelsT1, T2, labelsT2, labelsR)
  return RR
end

function show(io::IO,
                   mime::MIME"text/plain",
                   T::EmptyTensor)
  summary(io, T)
  println(io)
end

