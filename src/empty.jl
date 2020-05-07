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

Base.copy(S::Empty) = S

Base.isempty(::Empty) = true

Base.size(::Empty) = 0

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

Base.size(::EmptyTensor) = 0

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

function Base.setindex(T::EmptyTensor{<:Number, Any},
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

