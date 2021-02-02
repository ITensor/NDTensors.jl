
insertblock!!(T::EmptyTensor{<: Number, N}, block) where {N} =
  insertblock(T, block)

setindex!!(T::EmptyTensor, x, I...) = setindex(T, x, I...)

# Version of contraction where output storage is empty
function contract!!(R::EmptyTensor, labelsR::NTuple,
                    T1::Tensor, labelsT1::NTuple,
                    T2::Tensor, labelsT2::NTuple)
  return contract(T1, labelsT1, T2, labelsT2, labelsR)
end

function contract!!(R_labelsR::Tuple{<: EmptyTensor, <: NTuple},
                    T1_labelsT1::Tuple{<: Tensor, <: NTuple},
                    T2_labelsT2::Tuple{<: Tensor, <: NTuple})
  return contract!!(R_labelsR..., T1_labelsT1..., T2_labelsT2...)
end

