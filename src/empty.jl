
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

Base.copy(S::Empty) = S

Base.isempty(::Empty) = true

nnzblocks(::Empty) = 0

nnz(::Empty) = 0

function Base.complex(::Type{<: Empty{ElT, StoreT}}) where {ElT,
                                                            StoreT}
  return Empty{complex(ElT), complex(StoreT)}
end

function Base.complex(S::Empty)
  return complex(typeof(S))()
end

#Base.size(::Empty) = 0

function Base.show(io::IO,
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

Base.isempty(::EmptyTensor) = true

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
function Base.fill(::Type{<: Tensor{ElT,
                                    N,
                                    EStoreT,
                                    IndsT}}) where {ElT <: Number,
                                                    N,
                                                    EStoreT <: Empty{ElT,
                                                                     StoreT},
                                                    IndsT} where {StoreT}
  return Tensor{ElT, N, StoreT, IndsT}
end

function Base.zeros(T::TensorT) where {TensorT <: EmptyTensor}
  TensorR = fill(TensorT)
  return zeros(TensorR, inds(T))
end

function addblock(T::EmptyTensor{<: Number, N},
                  block::Block{N}) where {N}
  R = zeros(T)
  addblock!(R, block)
  return R
end

addblock!!(T::EmptyTensor{<: Number, N},
           block::Block{N}) where {N} = addblock(T, block)

Base.@propagate_inbounds function Base.setindex(T::EmptyTensor{<: Number, N},
                                                x::Number,
                                                I::Vararg{Int, N}) where {N}
  R = zeros(T)
  R[I...] = x
  return R
end

function Base.setindex(T::EmptyTensor{<: Number, Any},
                                      x::Number,
                                      I::Int...)
  error("Setting element of EmptyTensor with Any number of dimensions not defined")
end

setindex!!(T::EmptyTensor,
           x::Number,
           I::Int...) = setindex(T, x, I...)

function Base.show(io::IO,
                   mime::MIME"text/plain",
                   T::EmptyTensor)
  summary(io, T)
  println(io)
end

