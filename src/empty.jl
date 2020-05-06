export Empty,
       EmptyTensor

#
# Empty storage
#

struct Empty{ElT, StoreT <: TensorStorage} <: TensorStorage{ElT}
end

# Defaults to Dense
Empty{ElT}() where {ElT} = Empty{ElT, Dense{ElT, Vector{ElT}}}()

Empty() = Empty{Float64}()

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

# If no indices are provided, the tensor has Any number of
# indices
tensor(S::Empty, ::Nothing) = Tensor{eltype(S), Any, typeof(S), Nothing}(nothing, S)

# From an EmptyTensor, return the closest Tensor type
function Base.fill(::Type{<:Tensor{ElT, N, EStoreT, IndsT}}) where {ElT,
                                                                    N,
                                                                    EStoreT <: Empty{ElT, StoreT},
                                                                    IndsT} where {StoreT}
  return Tensor{ElT, N, StoreT, IndsT}
end

Base.@propagate_inbounds function Base.setindex(T::EmptyTensor{<:Number, N},
                                                x::Number,
                                                I::Vararg{Int, N}) where {N}
  R = zeros(fill(typeof(T)), inds(T))
  R[I...] = x
  return R
end

setindex!!(T::EmptyTensor, x::Number, I::Int...) = setindex(T, x, I...)

function Base.show(io::IO,
                   mime::MIME"text/plain",
                   ::Empty)
  nothing
end

function Base.show(io::IO,
                   mime::MIME"text/plain",
                   T::EmptyTensor)
  summary(io, T)
  println(io)
end

