
abstract type TensorStorage{ElT} <: AbstractVector{ElT} end

data(S::TensorStorage) = S.data

Base.eltype(::TensorStorage{ElT}) where {ElT} = ElT

Base.eltype(::Type{<:TensorStorage{ElT}}) where {ElT} = ElT

Base.iterate(S::TensorStorage,args...) = iterate(data(S),args...)

dense(S::TensorStorage) = S

# This is necessary since for some reason inference doesn't work
# with the more general definition (eltype(Nothing) === Any)
Base.eltype(::TensorStorage{Nothing}) = Nothing

Base.length(S::TensorStorage) = length(data(S))

Base.size(S::TensorStorage) = size(data(S))

Base.@propagate_inbounds Base.getindex(S::TensorStorage,
                                       i::Integer) = data(S)[i]
Base.@propagate_inbounds Base.setindex!(S::TensorStorage,v,
                                        i::Integer) = (setindex!(data(S),v,i); S)

# Needed for passing Tensor{T,2} to BLAS/LAPACK
function Base.unsafe_convert(::Type{Ptr{ElT}},
                             T::TensorStorage{ElT}) where {ElT}
  return Base.unsafe_convert(Ptr{ElT},data(T))
end

# This may need to be overloaded, since storage types
# often have other fields besides data

Base.conj!(S::TensorStorage) = (conj!(data(S)); return S)

function Base.conj(S::T;always_copy = false) where {T<:TensorStorage} 
  if always_copy
    return conj!(copy(S))
  end
  return T(conj(data(S)))
end

function Base.complex(S::T) where {T <: TensorStorage}
  return complex(T)(complex(data(S)))
end

Base.copyto!(S1::TensorStorage,
             S2::TensorStorage) = (copyto!(data(S1), data(S2)); S1)

Random.randn!(S::TensorStorage) = (randn!(data(S)); S)

Base.fill!(S::TensorStorage, v) = (fill!(data(S), v); S)

LinearAlgebra.rmul!(S::TensorStorage,
                    v::Number) = (rmul!(data(S), v); S)

scale!(S::TensorStorage, v::Number) = rmul!(S, v)

LinearAlgebra.norm(S::TensorStorage) = norm(data(S))

Base.convert(::Type{T},S::T) where {T <: TensorStorage} = S

blockoffsets(S::TensorStorage) = S.blockoffsets

"""
nzblocks(T::TensorStorage)

Return a vector of the non-zero blocks of the BlockSparse storage.
"""
nzblocks(T::TensorStorage) = nzblocks(blockoffsets(T))

eachnzblock(T::TensorStorage) = eachnzblock(blockoffsets(T))

nnzblocks(S::TensorStorage) = length(blockoffsets(S))
nnz(S::TensorStorage) = length(S)

offset(S::TensorStorage, block) = offset(blockoffsets(S), block)

