
# Version that may overwrite the result or promote
# and return the result
# TODO: move to tensor.jl?
function permutedims!!(R::Tensor, T::Tensor,
                       perm::NTuple{N,Int},
                       f::Function=(r,t)->t) where {N}
  #RA = array(R)
  #TA = array(T)
  RA = ReshapedArray(data(R), dims(R), ())
  TA = ReshapedArray(data(T), dims(T), ())
  if !is_trivial_permutation(perm)
    @strided RA .= f.(RA, permutedims(TA, perm))
  else
    # TODO: specialize for specific functions
    RA .= f.(RA, TA)
  end
  return R
end

function outer!!(R::Tensor,
                 T1::Tensor,
                 T2::Tensor)
  outer!(R,T1,T2)
  return R
end

# Move to tensor.jl? Overload this function
# for immutable storage types
function _contract!!(R::Tensor, labelsR,
                     T1::Tensor, labelsT1,
                     T2::Tensor, labelsT2,
                     α::Number=1, β::Number=0)
  if α ≠ 1 || β ≠ 0
    contract!(R, labelsR,
              T1, labelsT1,
              T2, labelsT2,
              α, β)
  else
    contract!(R, labelsR,
              T1, labelsT1,
              T2, labelsT2)
  end
  return R
end

# Move to tensor.jl? Is this generic for all storage types?
function contract!!(R::DenseTensor{<:Number,NR},
                    labelsR::NTuple{NR},
                    T1::DenseTensor{<:Number,N1},
                    labelsT1::NTuple{N1},
                    T2::DenseTensor{<:Number,N2},
                    labelsT2::NTuple{N2},
                    α::Number=1, β::Number=0) where {NR,N1,N2}
  if (N1 ≠ 0) && (N2 ≠ 0) && (N1 + N2 == NR)
    # Outer product
    (α ≠ 1 || β ≠ 0) &&
      error("contract!! not yet implemented for outer product tensor contraction with non-trivial α and β")
    # TODO: permute T1 and T2 appropriately first (can be more efficient
    # then permuting the result of T1⊗T2)
    # TODO: implement the in-place version directly
    R = outer!!(R, T1, T2)
    labelsRp = (labelsT1..., labelsT2...)
    perm = getperm(labelsR, labelsRp)
    if !is_trivial_permutation(perm)
      Rp = reshape(R, (inds(T1)..., inds(T2)...))
      R = permutedims!!(R, copy(Rp), perm)
    end
  else
    if α ≠ 1 || β ≠ 0
      R = _contract!!(R, labelsR,
                      T1, labelsT1,
                      T2, labelsT2,
                      α, β)
    else
      R = _contract!!(R, labelsR,
                      T1, labelsT1,
                      T2, labelsT2)
    end
  end
  return R
end

