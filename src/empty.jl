
#
# Empty Number
#
# Represents a number that can be set to any type.
#

struct EmptyNumber <: Number end

#
# Empty storage
#

struct EmptyStorage{ElT,StoreT<:TensorStorage} <: TensorStorage{ElT} end

# Get the EmptyStorage version of the TensorStorage
function emptytype(::Type{StoreT}) where {StoreT}
  return EmptyStorage{eltype(StoreT),StoreT}
end

empty(::Type{StoreT}) where {StoreT} = emptytype(StoreT)()

# Defaults to Dense
function EmptyStorage(::Type{ElT}) where {ElT}
  return emptytype(Dense{ElT,Vector{ElT}})()
end

EmptyStorage() = EmptyStorage(Float64)

similar(S::EmptyStorage) = S
similar(S::EmptyStorage, ::Type{ElT}) where {ElT} = empty(similartype(fulltype(S), ElT))

copy(S::EmptyStorage) = S

isempty(::EmptyStorage) = true

nnzblocks(::EmptyStorage) = 0

nnz(::EmptyStorage) = 0

Base.real(::Type{<:EmptyStorage{ElT,StoreT}}) where {ElT,StoreT} = EmptyStorage{real(ElT),real(StoreT)}

Base.real(S::EmptyStorage) = real(typeof(S))()

function complex(::Type{<:EmptyStorage{ElT,StoreT}}) where {ElT,StoreT}
  return EmptyStorage{complex(ElT),complex(StoreT)}
end

complex(S::EmptyStorage) = complex(typeof(S))()

#size(::EmptyStorage) = 0

function show(io::IO, mime::MIME"text/plain", S::EmptyStorage)
  return println(io, typeof(S))
end

#
# EmptyTensor (Tensor using EmptyStorage storage)
#

const EmptyTensor{ElT,N,StoreT,IndsT} = Tensor{ElT,N,StoreT,IndsT} where {StoreT<:EmptyStorage}

# XXX TODO: add bounds checking
getindex(T::EmptyTensor, I::Integer...) = zero(eltype(T))
getindex(T::EmptyTensor{EmptyNumber}, I::Integer...) = zero(Float64)

similar(T::EmptyTensor, inds::Tuple) = setinds(T, inds)
function similar(T::EmptyTensor, ::Type{ElT}) where {ElT<:Number}
  return tensor(similar(storage(T), ElT), inds(T))
end

function permutedims!!(
  R::EmptyTensor, T::EmptyTensor, perm::Tuple, f::Function=(r, t) -> t
)
  return R
end

function randn!!(T::EmptyTensor)
  Tf = similar(fulltype(T), inds(T))
  randn!(Tf)
  return Tf
end

# Default to Float64
function randn!!(T::EmptyTensor{EmptyNumber})
  return randn!!(similar(T, Float64))
end

function _fill!!(::Type{ElT}, T::EmptyTensor, α::Number) where {ElT}
  Tf = similar(fulltype(T), ElT, inds(T))
  fill!(Tf, α)
  return Tf
end

fill!!(T::EmptyTensor, α::Number) = _fill!!(eltype(T), T, α)

# Determine the element type from the number you are filling with
fill!!(T::EmptyTensor{EmptyNumber}, α::Number) = _fill!!(eltype(α), T, α)

isempty(::EmptyTensor) = true

function EmptyTensor(::Type{ElT}, inds) where {ElT<:Number}
  return tensor(EmptyStorage(ElT), inds)
end

function EmptyTensor(::Type{StoreT}, inds) where {StoreT<:TensorStorage}
  return tensor(empty(StoreT), inds)
end

function EmptyBlockSparseTensor(::Type{ElT}, inds) where {ElT<:Number}
  StoreT = BlockSparse{ElT,Vector{ElT},length(inds)}
  return EmptyTensor(StoreT, inds)
end

fulltype(::Type{EmptyStorage{ElT,StoreT}}) where {ElT,StoreT} = StoreT
fulltype(T::EmptyStorage) = fulltype(typeof(T))

fulltype(T::Tensor) = fulltype(typeof(T))

# From an EmptyTensor, return the closest Tensor type
function fulltype(::Type{TensorT}) where {TensorT<:Tensor}
  return Tensor{eltype(TensorT),ndims(TensorT),fulltype(storetype(TensorT)),indstype(TensorT)}
end

# TODO: make these more general, move to tensorstorage.jl
datatype(::Type{<:Dense{<:Any,DataT}}) where {DataT} = DataT

# TODO: make these more general, move to tensorstorage.jl
# TODO: rename `similartype`
similartype(::Type{<:Vector}, ::Type{ElT}) where {ElT} = Vector{ElT}

function similartype(StoreT::Type{<:Dense{EmptyNumber}}, ::Type{ElT}) where {ElT}
  return Dense{ElT,similartype(datatype(StoreT), ElT)}
end

function fulltype(
  ::Type{ElR},
  ::Type{<:Tensor{ElT,N,EStoreT,IndsT}}
) where {ElR,ElT<:Number,N,EStoreT<:EmptyStorage{ElT,StoreT},IndsT} where {StoreT}
  return Tensor{ElR,N,similartype(StoreT,ElR),IndsT}
end

function zeros(T::TensorT) where {TensorT<:EmptyTensor}
  TensorR = fulltype(TensorT)
  return zeros(TensorR, inds(T))
end

function zeros(::Type{ElT}, T::TensorT) where {ElT,TensorT<:EmptyTensor}
  TensorR = fulltype(ElT, TensorT)
  return zeros(TensorR, inds(T))
end

function insertblock(T::EmptyTensor{<:Number,N}, block) where {N}
  R = zeros(T)
  insertblock!(R, Block(block))
  return R
end

insertblock!!(T::EmptyTensor{<:Number,N}, block) where {N} = insertblock(T, block)

# Special case with element type of EmptyNumber: storage takes the type
# of the input
@propagate_inbounds function _setindex(T::EmptyTensor{EmptyNumber}, x, I...)
  R = zeros(typeof(x), T)
  R[I...] = x
  return R
end

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
function contract!!(
  R::EmptyTensor,
  labelsR,
  T1::Tensor,
  labelsT1,
  T2::Tensor,
  labelsT2,
)
  RR = contract(T1, labelsT1, T2, labelsT2, labelsR)
  return RR
end

# For ambiguity with versions in combiner.jl
function contract!!(
  R::EmptyTensor,
  labelsR,
  T1::CombinerTensor,
  labelsT1,
  T2::Tensor,
  labelsT2,
)
  RR = contract(T1, labelsT1, T2, labelsT2, labelsR)
  return RR
end

# For ambiguity with versions in combiner.jl
function contract!!(
  R::EmptyTensor,
  labelsR,
  T1::Tensor,
  labelsT1,
  T2::CombinerTensor,
  labelsT2,
)
  RR = contract(T1, labelsT1, T2, labelsT2, labelsR)
  return RR
end

function show(io::IO, mime::MIME"text/plain", T::EmptyTensor)
  summary(io, T)
  return println(io)
end
