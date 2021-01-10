
import Base:
  # Types
  Array,
  CartesianIndex,
  IndexStyle,
  Tuple,
  # Symbols
  +,
  *,
  # Methods
  checkbounds,
  complex,
  convert,
  conj,
  copy,
  copyto!,
  eachindex,
  eltype,
  fill,
  fill!,
  getindex,
  hash,
  isempty,
  iterate,
  length,
  ndims,
  permutedims,
  permutedims!,
  promote_rule,
  randn,
  reshape,
  setindex,
  setindex!,
  show,
  size,
  similar,
  stride,
  strides,
  summary,
  to_indices,
  unsafe_convert,
  view,
  zeros

import Base.Broadcast:
  Broadcasted,
  BroadcastStyle

import TupleTools:
  isperm

