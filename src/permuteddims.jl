
struct PermutedDims{TensorT <: Tensor, PermT <: Tuple}
  parent::TensorT
  perm::PermT
end

