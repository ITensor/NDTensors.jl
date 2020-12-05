#
# BlockSparseTensor (Tensor using BlockSparse storage)
#

const BlockSparseTensor{ElT,N,StoreT,IndsT} = Tensor{ElT,N,StoreT,IndsT} where {StoreT<:BlockSparse}

# Special version for BlockSparseTensor
# Generic version doesn't work since BlockSparse us parametrized by
# the Tensor order
function similar_type(::Type{<:Tensor{ElT,NT,<:BlockSparse{ElT,VecT},<:Any}},
                      ::Type{IndsR}) where {NT,ElT,VecT,IndsR}
  NR = length(IndsR)
  return Tensor{ElT,NR,BlockSparse{ElT,VecT,NR},IndsR}
end

function similar_type(::Type{<:Tensor{ElT,NT,<:BlockSparse{ElT,VecT},<:Any}},
                      ::Type{IndsR}) where {NT,ElT,VecT,IndsR<:NTuple{NR}} where {NR}
  return Tensor{ElT,NR,BlockSparse{ElT,VecT,NR},IndsR}
end

function BlockSparseTensor(::Type{ElT},
                           ::UndefInitializer,
                           boffs::BlockOffsets{N},
                           inds) where {ElT<:Number,N}
  nnz_tot = nnz(boffs,inds)
  storage = BlockSparse(ElT,undef,boffs,nnz_tot)
  return tensor(storage,inds)
end

function BlockSparseTensor(::UndefInitializer,
                           blockoffsets::BlockOffsets{N},
                           inds) where {N}
  return BlockSparseTensor(Float64,undef,blockoffsets,inds)
end

function BlockSparseTensor(::Type{ElT},
                           blockoffsets::BlockOffsets{N},
                           inds) where {ElT<:Number,N}
  nnz_tot = nnz(blockoffsets,inds)
  storage = BlockSparse(ElT,blockoffsets,nnz_tot)
  return tensor(storage,inds)
end

function BlockSparseTensor(blockoffsets::BlockOffsets,
                           inds)
  return BlockSparseTensor(Float64,blockoffsets,inds)
end


"""
BlockSparseTensor(::UndefInitializer,
                  blocks::Vector{Block{N}},
                  inds)

Construct a block sparse tensor with uninitialized memory
from indices and locations of non-zero blocks.
"""
BlockSparseTensor(::UndefInitializer,
                  blocks::Blocks,
                  inds) = BlockSparseTensor(Float64,undef,blocks,inds)

function BlockSparseTensor(::Type{ElT},
                           ::UndefInitializer,
                           blocks::Blocks,
                           inds) where {ElT}
  boffs,nnz = blockoffsets(blocks,inds)
  storage = BlockSparse(ElT,undef,boffs,nnz)
  return tensor(storage,inds)
end

#function BlockSparseTensor(::UndefInitializer,
#                           blocks::Blocks{N},
#                           inds::Vararg{DimT,N}) where {DimT,N}
#  return BlockSparseTensor(undef,blocks,inds)
#end

"""
BlockSparseTensor(inds)

Construct a block sparse tensor with no blocks.
"""
function BlockSparseTensor(inds)
  return BlockSparseTensor(BlockOffsets{length(inds)}(),inds)
end

function BlockSparseTensor(::Type{ElT},
                           inds) where {ElT<:Number,N}
  return BlockSparseTensor(ElT,BlockOffsets{length(inds)}(),inds)
end

"""
BlockSparseTensor(inds)

Construct a block sparse tensor with no blocks.
"""
function BlockSparseTensor(inds::Vararg{DimT,N}) where {DimT,N}
  return BlockSparseTensor(BlockOffsets{N}(),inds)
end

"""
BlockSparseTensor(blocks::Vector{Block{N}},
                  inds)

Construct a block sparse tensor with the specified blocks.
Defaults to setting structurally non-zero blocks to zero.
"""
BlockSparseTensor(blocks::Blocks,
                  inds) = BlockSparseTensor(Float64,blocks,inds)

function BlockSparseTensor(::Type{ElT},
                           blocks::Blocks,
                           inds) where {ElT}
  boffs,nnz = blockoffsets(blocks,inds)
  storage = BlockSparse(ElT,boffs,nnz)
  return tensor(storage,inds)
end

#function BlockSparseTensor(blocks::Vector{Block{N}},
#                           inds) where {N}
#  blockoffsets,nnz = blockoffsets(blocks,inds)
#  storage = BlockSparse(blockoffsets,nnz)
#  return tensor(storage,inds)
#end

function Base.randn(::Type{ <: BlockSparseTensor{ElT, N}},
                    blocks::Blocks{N},
                    inds) where {ElT, N}
  boffs, nnz = blockoffsets(blocks, inds)
  storage = randn(BlockSparse{ElT}, boffs, nnz)
  return tensor(storage, inds)
end

function randomBlockSparseTensor(::Type{ElT},
                                 blocks::Blocks{N},
                                 inds) where {ElT, N}
  return randn(BlockSparseTensor{ElT, N}, blocks, inds)
end

function randomBlockSparseTensor(blocks::Blocks,
                                 inds)
  return randomBlockSparseTensor(Float64, blocks, inds)
end

"""
BlockSparseTensor(blocks::Vector{Block{N}},
                  inds::BlockDims...)

Construct a block sparse tensor with the specified blocks.
Defaults to setting structurally non-zero blocks to zero.
"""
function BlockSparseTensor(blocks::Blocks{N},
                           inds::Vararg{BlockDim,N}) where {DimT,N}
  return BlockSparseTensor(blocks,inds)
end

function Base.similar(::BlockSparseTensor{ElT,N},
                      blockoffsets::BlockOffsets{N},
                      inds) where {ElT,N}
  return BlockSparseTensor(ElT,undef,blockoffsets,inds)
end

function Base.similar(::Type{<:BlockSparseTensor{ElT,N}},
                      blockoffsets::BlockOffsets{N},
                      inds) where {ElT,N}
  return BlockSparseTensor(ElT,undef,blockoffsets,inds)
end

function Base.zeros(::BlockSparseTensor{ElT,N},
                    blockoffsets::BlockOffsets{N},
                    inds) where {ElT,N}
  return BlockSparseTensor(ElT,blockoffsets,inds)
end

function Base.zeros(::Type{<:BlockSparseTensor{ElT,N}},
                    blockoffsets::BlockOffsets{N},
                    inds) where {ElT,N}
  return BlockSparseTensor(ElT,blockoffsets,inds)
end

function Base.zeros(::BlockSparseTensor{ElT, N},
                    inds) where {ElT, N}
  return BlockSparseTensor(ElT, inds)
end

function Base.zeros(::Type{<: BlockSparseTensor{ElT, N}},
                    inds) where {ElT, N}
  return BlockSparseTensor(ElT, inds)
end

# Basic functionality for AbstractArray interface
Base.IndexStyle(::Type{<:BlockSparseTensor}) = IndexCartesian()

# Get the CartesianIndices for the range of indices
# of the specified
function blockindices(T::BlockSparseTensor{ElT,N},
                      block) where {ElT,N}
  return CartesianIndex(blockstart(T,block)):CartesianIndex(blockend(T,block))
end

"""
indexoffset(T::BlockSparseTensor,i::Int...) -> offset,block,blockoffset

Get the offset in the data of the specified
CartesianIndex. If it falls in a block that doesn't
exist, return nothing for the offset.
Also returns the block the index is found in and the offset
within the block.
"""
function indexoffset(T::BlockSparseTensor{ElT,N},
                     i::Vararg{Int,N}) where {ElT,N}
  index_within_block,block = blockindex(T,i...)
  block_dims = blockdims(T,block)
  offset_within_block = LinearIndices(block_dims)[CartesianIndex(index_within_block)]
  offset_of_block = offset(T,block)
  offset_of_i = isnothing(offset_of_block) ? nothing : offset_of_block+offset_within_block
  return offset_of_i,block,offset_within_block
end

# TODO: Add a checkbounds
# TODO: write this nicer in terms of blockview?
#       Could write: 
#       block,index_within_block = blockindex(T,i...)
#       return blockview(T,block)[index_within_block]
Base.@propagate_inbounds function Base.getindex(T::BlockSparseTensor{ElT,N},
                                                i::Vararg{Int,N}) where {ElT,N}
  offset,_ = indexoffset(T,i...)
  isnothing(offset) && return zero(ElT)
  return store(T)[offset]
end

Base.@propagate_inbounds function Base.getindex(T::BlockSparseTensor{ElT,0}) where {ElT}
  nnzblocks(T) == 0 && return zero(ElT)
  return store(T)[]
end

# These may not be valid if the Tensor has no blocks
#Base.@propagate_inbounds Base.getindex(T::BlockSparseTensor{<:Number,1},ind::Int) = store(T)[ind]

#Base.@propagate_inbounds Base.getindex(T::BlockSparseTensor{<:Number,0}) = store(T)[1]

# Add the specified block to the BlockSparseTensor
# Insert it such that the blocks remain ordered.
# Defaults to adding zeros.
# Returns the offset of the new block added.
# XXX rename to insertblock!, no need to return offset
function insertblock_offset!(T::BlockSparseTensor{ElT, N},
                             newblock::Block{N}) where {ElT, N}
  newdim = blockdim(T,newblock)
  newoffset = nnz(T)
  blockoffsets(T)[newblock] = newoffset
  # Insert new block into data
  splice!(data(store(T)), newoffset+1:newoffset, zeros(ElT,newdim))
  return newoffset
end

function insertblock!(T::BlockSparseTensor{<: Number, N},
                      block::Block{N}) where {N}
  insertblock_offset!(T, block)
  return T
end

# TODO: Add a checkbounds
Base.@propagate_inbounds function Base.setindex!(T::BlockSparseTensor{ElT,N},
                                                 val,
                                                 i::Vararg{Int,N}) where {ElT,N}
  offset,block,offset_within_block = indexoffset(T,i...)
  if isnothing(offset)
    offset_of_block = insertblock_offset!(T, block)
    offset = offset_of_block+offset_within_block
  end
  store(T)[offset] = val
  return T
end

blockview(T::BlockSparseTensor, blockT::Block) = blockview(T, BlockOffset(blockT, offset(T, blockT)))

function blockview(T::BlockSparseTensor,
                   bof::BlockOffset)
  blockT, offsetT = bof
  blockdimsT = blockdims(T, blockT)
  blockdimT = prod(blockdimsT)
  dataTslice = @view data(store(T))[offsetT+1:offsetT+blockdimT]
  return tensor(Dense(dataTslice), blockdimsT)
end

# convert to Dense
function dense(T::TensorT) where {TensorT<:BlockSparseTensor}
  R = zeros(dense(TensorT), inds(T))
  for (block,offset) in blockoffsets(T)
    # TODO: make sure this assignment is efficient
    R[blockindices(T,block)] = blockview(T,block)
  end
  return R
end

#
# Operations
#

# TODO: extend to case with different block structures
function Base.:+(T1::BlockSparseTensor,T2::BlockSparseTensor)
  inds(T1) ≠ inds(T2) && error("Cannot add block sparse tensors with different block structure")  
  return tensor(store(T1)+store(T2),inds(T1))
end

function Base.permutedims(T::BlockSparseTensor{<:Number,N},
                          perm::NTuple{N,Int}) where {N}
  blockoffsetsR,indsR = permutedims(blockoffsets(T),inds(T),perm)
  R = similar(T,blockoffsetsR,indsR)
  permutedims!(R,T,perm)
  return R
end

function _permute_combdims(combdims::NTuple{NC,Int},
                           perm::NTuple{NP,Int}) where {NC,NP}
  res = MVector{NC,Int}(undef)
  iperm = invperm(perm)
  for i in 1:NC
    res[i] = iperm[combdims[i]]
  end
  return Tuple(res)
end

#
# These are functions to help with combining and uncombining
#

# Note that combdims is expected to be contiguous and ordered
# smallest to largest
function combine_dims(blocks::Blocks{N},
                      inds,
                      combdims::NTuple{NC,Int}) where {N,NC}
  nblcks = nblocks(inds,combdims)
  blocks_comb = Blocks{N-NC+1}(undef,nnzblocks(blocks))
  for (i,block) in enumerate(blocks)
    blocks_comb[i] = combine_dims(block,inds,combdims)
  end
  return blocks_comb
end

function combine_dims(block::Block,
                      inds,
                      combdims::NTuple{NC,Int}) where {NC}
  nblcks = nblocks(inds,combdims)
  slice = getindices(block,combdims)
  slice_comb = LinearIndices(nblcks)[slice...]
  block_comb = deleteat(block,combdims)
  block_comb = insertafter(block_comb,tuple(slice_comb),minimum(combdims)-1)
  return block_comb
end

# In the dimension dim, permute the blocks
function perm_blocks(blocks::Blocks{N},
                     dim::Int,
                     perm) where {N}
  blocks_perm = Blocks{N}(undef,nnzblocks(blocks))
  iperm = invperm(perm)
  for (i,block) in enumerate(blocks)
    blocks_perm[i] = setindex(block,iperm[block[dim]],dim)
  end
  return blocks_perm
end

# In the dimension dim, permute the block
function perm_block(block::Block,
                    dim::Int,
                    perm) where {N}
  iperm = invperm(perm)
  return setindex(block,iperm[block[dim]],dim)
end

# In the dimension dim, combine the specified blocks
function combine_blocks(blocks::Blocks,
                        dim::Int,
                        blockcomb::Vector{Int})
  blocks_comb = copy(blocks)
  nnz_comb = nnzblocks(blocks)
  for (i,block) in enumerate(blocks)
    dimval = block[dim]
    blocks_comb[i] = setindex(block,blockcomb[dimval],dim)
  end
  unique!(blocks_comb)
  return blocks_comb
end

function permutedims_combine_output(T::BlockSparseTensor{ElT,N},
                                    is,
                                    perm::NTuple{N,Int},
                                    combdims::NTuple{NC,Int},
                                    blockperm::Vector{Int},
                                    blockcomb::Vector{Int}) where {ElT,N,NC}
  # Permute the indices
  indsT = inds(T)
  inds_perm = permute(indsT,perm)

  # Now that the indices are permuted, compute
  # which indices are now combined
  combdims_perm = sort(_permute_combdims(combdims,perm))

  # Permute the nonzero blocks (dimension-wise)
  blocks = nzblocks(T)
  blocks_perm = permutedims(blocks,perm)

  # Combine the nonzero blocks (dimension-wise)
  blocks_perm_comb = combine_dims(blocks_perm,inds_perm,combdims_perm)

  # Permute the blocks (within the newly combined dimension)
  comb_ind_loc = minimum(combdims_perm)
  blocks_perm_comb = perm_blocks(blocks_perm_comb,comb_ind_loc,blockperm)
  blocks_perm_comb = sort(blocks_perm_comb;lt=isblockless)

  # Combine the blocks (within the newly combined and permuted dimension)
  blocks_perm_comb = combine_blocks(blocks_perm_comb,comb_ind_loc,blockcomb)

  return BlockSparseTensor(ElT,blocks_perm_comb,is)
end

function permutedims_combine(T::BlockSparseTensor{ElT,N},
                             is,
                             perm::NTuple{N,Int},
                             combdims::NTuple{NC,Int},
                             blockperm::Vector{Int},
                             blockcomb::Vector{Int}) where {ElT,N,NC}

  R = permutedims_combine_output(T,is,perm,combdims,blockperm,blockcomb)

  # Permute the indices
  inds_perm = permute(inds(T),perm)

  # Now that the indices are permuted, compute
  # which indices are now combined
  combdims_perm = sort(_permute_combdims(combdims,perm))
  comb_ind_loc = minimum(combdims_perm)

  # Determine the new index before combining
  inds_to_combine = getindices(inds_perm,combdims_perm)
  ind_comb = ⊗(inds_to_combine...)
  ind_comb = permuteblocks(ind_comb,blockperm)

  for bof in blockoffsets(T)
    Tb = blockview(T,bof)
    b = nzblock(bof)
    b_perm = permute(b,perm)
    b_perm_comb = combine_dims(b_perm,inds_perm,combdims_perm)
    b_perm_comb = perm_block(b_perm_comb,comb_ind_loc,blockperm)
    b_in_combined_dim = b_perm_comb[comb_ind_loc]
    new_b_in_combined_dim = blockcomb[b_in_combined_dim]
    offset = 0
    pos_in_new_combined_block = 1
    while b_in_combined_dim-pos_in_new_combined_block > 0 && 
            blockcomb[b_in_combined_dim-pos_in_new_combined_block] == new_b_in_combined_dim
      offset += blockdim(ind_comb,b_in_combined_dim-pos_in_new_combined_block)
      pos_in_new_combined_block += 1
    end
    b_new = setindex(b_perm_comb,new_b_in_combined_dim,comb_ind_loc)

    Rb_total = blockview(R,b_new)
    dimsRb_tot = dims(Rb_total)
    subind = ntuple(i->i==comb_ind_loc ? range(1+offset,stop=offset+blockdim(ind_comb,b_in_combined_dim)) : range(1,stop=dimsRb_tot[i]),N-NC+1)
    Rb = @view array(Rb_total)[subind...]

    # XXX Are these equivalent?
    #Tb_perm = permutedims(Tb,perm)
    #copyto!(Rb,Tb_perm)

    # XXX Not sure what this was for
    Rb = reshape(Rb,permute(dims(Tb),perm))
    Tbₐ = convert(Array, Tb)
    @strided Rb .= permutedims(Tbₐ, perm)
  end

  return R
end

# TODO: optimize by avoiding findfirst
function _number_uncombined(blockval::Int,
                            blockcomb::Vector{Int})
  if blockval == blockcomb[end]
    return length(blockcomb)-findfirst(==(blockval),blockcomb)+1
  end
  return findfirst(==(blockval+1),blockcomb)-findfirst(==(blockval),blockcomb)
end

# TODO: optimize by avoiding findfirst
function _number_uncombined_shift(blockval::Int,
                                  blockcomb::Vector{Int})
  if blockval == 1
    return 0
  end
  ncomb_shift = 0
  for i = 1:blockval-1
    ncomb_shift += findfirst(==(i+1),blockcomb) - findfirst(==(i),blockcomb) - 1
  end
  return ncomb_shift
end

# Uncombine the blocks along the dimension dim
# according to the pattern in blockcomb (for example, blockcomb
# is [1,2,2,3] and dim = 2, so the blocks (1,2),(2,3) get
# split into (1,2),(1,3),(2,4))
function uncombine_blocks(blocks::Blocks{N},
                          dim::Int,
                          blockcomb::Vector{Int}) where {N}
  blocks_uncomb = Blocks{N}()
  ncomb_tot = 0
  for i in 1:length(blocks)
    block = blocks[i]
    blockval = block[dim]
    ncomb = _number_uncombined(blockval,blockcomb)
    ncomb_shift = _number_uncombined_shift(blockval,blockcomb)
    push!(blocks_uncomb,setindex(block,blockval+ncomb_shift,dim))
    for j in 1:ncomb-1
      push!(blocks_uncomb,setindex(block,blockval+ncomb_shift+j,dim))
    end
  end
  return blocks_uncomb
end

function uncombine_block(block::Block{N},
                         dim::Int,
                         blockcomb::Vector{Int}) where {N}
  blocks_uncomb = Blocks{N}()
  ncomb_tot = 0
  blockval = block[dim]
  ncomb = _number_uncombined(blockval,blockcomb)
  ncomb_shift = _number_uncombined_shift(blockval,blockcomb)
  push!(blocks_uncomb,setindex(block,blockval+ncomb_shift,dim))
  for j in 1:ncomb-1
    push!(blocks_uncomb,setindex(block,blockval+ncomb_shift+j,dim))
  end
  return blocks_uncomb
end

function uncombine_output(T::BlockSparseTensor{ElT,N},
                          is,
                          combdim::Int,
                          blockperm::Vector{Int},
                          blockcomb::Vector{Int}) where {ElT<:Number,N}
  ind_uncomb_perm = ⊗(setdiff(is,inds(T))...)
  inds_uncomb_perm = insertat(inds(T),ind_uncomb_perm,combdim)
  # Uncombine the blocks of T
  blocks_uncomb = uncombine_blocks(nzblocks(T),combdim,blockcomb)
  blocks_uncomb_perm = perm_blocks(blocks_uncomb,combdim,invperm(blockperm))
  boffs_uncomb_perm,nnz_uncomb_perm = blockoffsets(blocks_uncomb_perm,inds_uncomb_perm)
  T_uncomb_perm = tensor(BlockSparse(ElT,boffs_uncomb_perm,nnz_uncomb_perm),inds_uncomb_perm)
  R = reshape(T_uncomb_perm,is)
  return R
end

function Base.reshape(blockT::Block{NT},
                      indsT,
                      indsR) where {NT}
  nblocksT = nblocks(indsT)
  nblocksR = nblocks(indsR)
  blockR = Tuple(CartesianIndices(nblocksR)[LinearIndices(nblocksT)[CartesianIndex(blockT)]])
  return blockR
end

function uncombine(T::BlockSparseTensor{<:Number,NT},
                   is,
                   combdim::Int,
                   blockperm::Vector{Int},
                   blockcomb::Vector{Int}) where {NT}
  NR = length(is)
  R = uncombine_output(T,is,combdim,blockperm,blockcomb)
  invblockperm = invperm(blockperm)

  # This is needed for reshaping the block
  # It is already calculated in uncombine_output, use it from there
  ind_uncomb_perm = ⊗(setdiff(is,inds(T))...)
  ind_uncomb = permuteblocks(ind_uncomb_perm,blockperm)
  # Same as inds(T) but with the blocks uncombined
  inds_uncomb = insertat(inds(T),ind_uncomb,combdim)
  inds_uncomb_perm = insertat(inds(T),ind_uncomb_perm,combdim)

  for bof in blockoffsets(T)
    b = nzblock(bof)
    Tb_tot = blockview(T,bof)
    dimsTb_tot = dims(Tb_tot)

    bs_uncomb = uncombine_block(b,combdim,blockcomb)

    offset = 0
    for i in 1:length(bs_uncomb)
      b_uncomb = bs_uncomb[i]
      b_uncomb_perm = perm_block(b_uncomb,combdim,invblockperm)
      b_uncomb_perm_reshape = reshape(b_uncomb_perm,inds_uncomb_perm,is)

      Rb = blockview(R,b_uncomb_perm_reshape)

      b_uncomb_in_combined_dim = b_uncomb_perm[combdim]

      start = offset+1
      stop = offset+blockdim(ind_uncomb_perm,b_uncomb_in_combined_dim)
      subind = ntuple(i->i==combdim ? range(start,stop=stop) : range(1,stop=dimsTb_tot[i]),NT)

      offset = stop

      Tb = @view array(Tb_tot)[subind...]

      # Alternative (but maybe slower):
      #copyto!(Rb,Tb)
 
      Rbₐ = convert(Array, Rb)
      Rbₐ = reshape(Rbₐ, size(Tb))
      @strided Rbₐ .= Tb
    end
  end
  return R
end

# TODO: handle case with different element types in R and T
#function permutedims!!(R::BlockSparseTensor{<:Number,N},
#                       T::BlockSparseTensor{<:Number,N},
#                       perm::NTuple{N,Int}) where {N}
#  blockoffsetsTp,indsTp = permute(blockoffsets(T),inds(T),perm)
#  if blockoffsetsTp == blockoffsets(R)
#    R = permutedims!(R,T,perm)
#    return R
#  end
#  R = similar(T,blockoffsetsTp,indsTp)
#  permutedims!(R,T,perm)
#  return R
#end

function Base.copyto!(R::BlockSparseTensor,
                      T::BlockSparseTensor)
  for bof in blockoffsets(T)
    copyto!(blockview(R, nzblock(bof)), blockview(T, bof))
  end
  return R
end

# TODO: handle case where:
# f(zero(ElR),zero(ElT)) != promote_type(ElR,ElT)
function permutedims!!(R::BlockSparseTensor{ElR,N},
                       T::BlockSparseTensor{ElT,N},
                       perm::NTuple{N,Int},
                       f::Function=(r,t)->t) where {ElR,ElT,N}
  # TODO: write a custom function for merging two sorted
  # lists with no repeated elements
  nzblocksRR = unique!(sort(vcat(nzblocks(R),permutedims(nzblocks(T),perm));lt=isblockless))
  RR = BlockSparseTensor(promote_type(ElR,ElT),nzblocksRR,inds(R))
  copyto!(RR,R)
  permutedims!(RR,T,perm,f)
  return RR
end

function Base.permutedims!(R::BlockSparseTensor{<:Number,N},
                           T::BlockSparseTensor{<:Number,N},
                           perm::NTuple{N,Int},
                           f::Function=(r,t)->t) where {N}
  for (blockT,_) in blockoffsets(T)
    # Loop over non-zero blocks of T/R
    Tblock = blockview(T,blockT)
    Rblock = blockview(R,permute(blockT,perm))
    permutedims!(Rblock,Tblock,perm,f)
  end
  return R
end

#function Base.permutedims!(R::BlockSparseTensor{<:Number,N},
#                           T::BlockSparseTensor{<:Number,N},
#                           perm::NTuple{N,Int},
#                           f::Function) where {N}
#  for (blockT,_) in blockoffsets(T)
#    # Loop over non-zero blocks of T/R
#    Tblock = blockview(T,blockT)
#    Rblock = blockview(R,permute(blockT,perm))
#    permutedims!(Rblock,Tblock,perm,f)
#  end
#  return R
#end

#
# Contraction
#

# TODO: complete this function: determine the output blocks from the input blocks
# Also, save the contraction list (which block-offsets contract with which),
# may not be generic with other contraction functions!
function contraction_output(T1::TensorT1,
                            T2::TensorT2,
                            indsR::IndsR) where {TensorT1<:BlockSparseTensor,
                                                 TensorT2<:BlockSparseTensor,
                                                 IndsR}
  TensorR = contraction_output_type(TensorT1,TensorT2,IndsR)
  return similar(TensorR,blockoffsetsR,indsR)
end

"""
find_matching_positions(t1,t2) -> t1_to_t2

In a tuple of length(t1), store the positions in t2
where the element of t1 is found. Otherwise, store 0
to indicate that the element of t1 is not in t2.

For example, for all t1[pos1] == t2[pos2], t1_to_t2[pos1] == pos2,
otherwise t1_to_t2[pos1] == 0.
"""
function find_matching_positions(t1,t2)
  t1_to_t2 = @MVector zeros(Int,length(t1))
  for pos1 in 1:length(t1)
    for pos2 in 1:length(t2)
      if t1[pos1] == t2[pos2]
        t1_to_t2[pos1] = pos2
      end
    end
  end
  return Tuple(t1_to_t2)
end

function contract_labels(labels1,labels2,labelsR)
  labels1_to_labels2 = find_matching_positions(labels1,labels2)
  labels1_to_labelsR = find_matching_positions(labels1,labelsR)
  labels2_to_labelsR = find_matching_positions(labels2,labelsR)
  return labels1_to_labels2,labels1_to_labelsR,labels2_to_labelsR
end

function are_blocks_contracted(block1::Block{N1},
                               block2::Block{N2},
                               labels1_to_labels2::NTuple{N1,Int}) where {N1,N2}
  for i1 in 1:N1
    i2 = labels1_to_labels2[i1]
    if i2 > 0
      # This dimension is contracted
      if block1[i1] != block2[i2]
        return false
      end
    end
  end
  return true
end

function contract_blocks(block1::Block{N1},
                         labels1_to_labelsR,
                         block2::Block{N2},
                         labels2_to_labelsR,
                         ::Val{NR}) where {N1,N2,NR}
  blockR = MVector(ntuple(_ -> 0, Val(NR)))
  for i1 in 1:N1
    iR = labels1_to_labelsR[i1]
    if iR > 0
      blockR[iR] = block1[i1]
    end
  end
  for i2 in 1:N2
    iR = labels2_to_labelsR[i2]
    if iR > 0
      blockR[iR] = block2[i2]
    end
  end
  return Tuple(blockR)    
end

function contract_blockoffsets(boffs1::BlockOffsets{N1},inds1,labels1,
                               boffs2::BlockOffsets{N2},inds2,labels2,
                               indsR,labelsR) where {N1,N2}
  NR = length(labelsR)
  ValNR = ValLength(labelsR)
  labels1_to_labels2,labels1_to_labelsR,labels2_to_labelsR = contract_labels(labels1,labels2,labelsR)
  blockoffsetsR = BlockOffsets{NR}()
  nnzR = 0
  contraction_plan = Tuple{NTuple{N1, Int}, NTuple{N2, Int}, NTuple{NR, Int}}[]
  for (pos1,(block1,offset1)) in enumerate(boffs1)
    for (pos2,(block2,offset2)) in enumerate(boffs2)
      if are_blocks_contracted(block1,block2,labels1_to_labels2)
        blockR = contract_blocks(block1,labels1_to_labelsR,
                                 block2,labels2_to_labelsR,
                                 ValNR)
        push!(contraction_plan, (block1, block2, blockR))
        if !haskey(blockoffsetsR, blockR)
          blockoffsetsR[blockR] = nnzR
          nnzR += blockdim(indsR, blockR)
        end
      end
    end
  end
  return blockoffsetsR,contraction_plan
end

function contraction_output(T1::TensorT1,
                            labelsT1,
                            T2::TensorT2,
                            labelsT2,
                            labelsR) where {TensorT1<:BlockSparseTensor,
                                            TensorT2<:BlockSparseTensor}
  indsR = contract_inds(inds(T1),labelsT1,inds(T2),labelsT2,labelsR)
  TensorR = contraction_output_type(TensorT1,TensorT2,typeof(indsR))
  blockoffsetsR,contraction_plan = contract_blockoffsets(blockoffsets(T1),inds(T1),labelsT1,
                                                         blockoffsets(T2),inds(T2),labelsT2,
                                                         indsR,labelsR)
  R = similar(TensorR,blockoffsetsR,indsR)
  return R,contraction_plan
end

function contract(T1::BlockSparseTensor{<:Any,N1},
                  labelsT1,
                  T2::BlockSparseTensor{<:Any,N2},
                  labelsT2,
                  labelsR = contract_labels(labelsT1,labelsT2)) where {N1,N2}
  R,contraction_plan = contraction_output(T1,labelsT1,T2,labelsT2,labelsR)
  R = contract!(R,labelsR,T1,labelsT1,T2,labelsT2,contraction_plan)
  return R
end

function contract!(R::BlockSparseTensor{ElR, NR},
                   labelsR,
                   T1::BlockSparseTensor{ElT1, N1},
                   labelsT1,
                   T2::BlockSparseTensor{ElT2, N2},
                   labelsT2,
                   contraction_plan) where {ElR, ElT1, ElT2,
                                            N1, N2, NR}
  #already_written_to = fill(false,nnzblocks(R))
  already_written_to = Dict{Block{NR}, Bool}()
  # In R .= α .* (T1 * T2) .+ β .* R
  α = one(ElR)
  for (pos1, pos2, posR) in contraction_plan
    blockT1 = blockview(T1,pos1)
    blockT2 = blockview(T2,pos2)
    blockR = blockview(R,posR)
    β = one(ElR)
    if !haskey(already_written_to, posR)
      already_written_to[posR] = true
      # Overwrite the block of R
      β = zero(ElR)
    end
    contract!(blockR,labelsR,
              blockT1,labelsT1,
              blockT2,labelsT2,
              α,β)
  end
  return R
end

const IntTuple = NTuple{N,Int} where N
const IntOrIntTuple = Union{Int,IntTuple}

function permute_combine(inds::IndsT,
                         pos::Vararg{IntOrIntTuple,N}) where {IndsT,N}
  IndT = eltype(IndsT)
  # Using SizedVector since setindex! doesn't
  # work for MVector when eltype not isbitstype
  newinds = SizedVector{N,IndT}(undef)
  for i ∈ 1:N
    pos_i = pos[i]
    newind_i = inds[pos_i[1]]
    for p in 2:length(pos_i)
      newind_i = newind_i ⊗ inds[pos_i[p]]
    end
    newinds[i] = newind_i
  end
  IndsR = similar_type(IndsT,Val{N})
  indsR = IndsR(Tuple(newinds))
  return indsR
end

"""
Indices are combined according to the grouping of the input,
for example (1,2),3 will combine the first two indices.
"""
function combine(inds::IndsT,
                 com::Vararg{IntOrIntTuple,N}) where {IndsT,N}
  IndT = eltype(IndsT)
  # Using SizedVector since setindex! doesn't
  # work for MVector when eltype not isbitstype
  newinds = SizedVector{N,IndT}(undef)
  i_orig = 1
  for i ∈ 1:N
    newind_i = inds[i_orig]
    i_orig += 1
    for p in 2:length(com[i])
      newind_i = newind_i ⊗ inds[i_orig]
      i_orig += 1
    end
    newinds[i] = newind_i
  end
  IndsR = similar_type(IndsT,Val{N})
  indsR = IndsR(Tuple(newinds))
  return indsR
end

function permute_combine(boffs::BlockOffsets,
                         inds::IndsT,
                         pos::Vararg{IntOrIntTuple,N}) where {IndsT,N}
  perm = flatten(pos...)
  boffsp,indsp = permutedims(boffs,inds,perm)
  indsR = combine(indsp,pos...)
  boffsR = reshape(boffsp,indsp,indsR)
  return boffsR,indsR
end

function Base.reshape(boffsT::BlockOffsets{NT},
                      indsT,
                      indsR) where {NT}
  NR = length(indsR)
  boffsR = BlockOffsets{NR}()
  nblocksT = nblocks(indsT)
  nblocksR = nblocks(indsR)
  for (blockT, offsetT) in boffsT
    blockR = Tuple(CartesianIndices(nblocksR)[LinearIndices(nblocksT)[CartesianIndex(blockT)]])
    boffsR[blockR] = offsetT
  end
  return boffsR
end

function Base.reshape(boffsT::BlockOffsets{NT},
                      blocksR::Vector{Block{NR}}) where {NR,NT}
  boffsR = BlockOffsets{NR}()
  # TODO: check blocksR is ordered and are properly reshaped
  # versions of the blocks of boffsT
  for (i,(blockT,offsetT)) in enumerate(boffsT)
    blockR = blocksR[i]
    boffsR[blockR] = offsetT
  end
  return boffsR
end

function Base.reshape(T::BlockSparse,
                      boffsR::BlockOffsets)
  return BlockSparse(data(T),boffsR)
end

function Base.reshape(T::BlockSparseTensor,
                      boffsR::BlockOffsets,
                      indsR)
  storeR = reshape(store(T),boffsR)
  return tensor(storeR,indsR)
end

function Base.reshape(T::BlockSparseTensor,
                      indsR)
  # TODO: add some checks that the block dimensions
  # are consistent (e.g. nnzblocks(T) == nnzblocks(R), etc.)
  boffsR = reshape(blockoffsets(T),inds(T),indsR)
  R = reshape(T,boffsR,indsR)
  return R
end

function permute_combine(T::BlockSparseTensor{ElT,NT,IndsT},
                         pos::Vararg{IntOrIntTuple,NR}) where {ElT,NT,IndsT,NR}
  boffsR,indsR = permute_combine(blockoffsets(T),inds(T),pos...)

  perm = flatten(pos...)

  length(perm)≠NT && error("Index positions must add up to order of Tensor ($NT)")
  isperm(perm) || error("Index positions must be a permutation")

  if !is_trivial_permutation(perm)
    Tp = permutedims(T,perm)
  else
    Tp = copy(T)
  end
  NR==NT && return Tp
  R = reshape(Tp,boffsR,indsR)
  return R
end

#
# Print block sparse tensors
#

#function Base.summary(io::IO,
#                      T::BlockSparseTensor{ElT,N}) where {ElT,N}
#  println(io,Base.dims2string(dims(T))," ",typeof(T))
#  for (dim,ind) in enumerate(inds(T))
#    println(io,"Dim $dim: ",ind)
#  end
#  println(io,"Number of nonzero blocks: ",nnzblocks(T))
#end

#function Base.summary(io::IO,
#                      T::BlockSparseTensor{ElT,N}) where {ElT,N}
#  println(io,typeof(T))
#  println(io,Base.dims2string(dims(T))," ",typeof(T))
#  for (dim,ind) in enumerate(inds(T))
#    println(io,"Dim $dim: ",ind)
#  end
#  println("Number of nonzero blocks: ",nnzblocks(T))
#end

function _range2string(rangestart::NTuple{N,Int},
                       rangeend::NTuple{N,Int}) where {N}
  s = ""
  for n in 1:N
    s = string(s,rangestart[n],":",rangeend[n])
    if n < N
      s = string(s,", ")
    end
  end
  return s
end

function Base.show(io::IO,
                   mime::MIME"text/plain",
                   T::BlockSparseTensor)
  summary(io,T)
  for (n, (block, _)) in enumerate(blockoffsets(T))
    blockdimsT = blockdims(T,block)
    println(io, "Block: ", block)
    println(io, " [", _range2string(blockstart(T,block),blockend(T,block)),"]")
    print_tensor(io, blockview(T,block))
    n < nnzblocks(T) && print(io, "\n\n")
  end
end

Base.show(io::IO, T::BlockSparseTensor) = show(io,MIME("text/plain"),T)

