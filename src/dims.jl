export dense,
       dims,
       dim,
       mindim,
       diaglength

# dim and dims are used in the Tensor interface, overload 
# base Dims here
dims(ds::Dims) = ds

# Generic dims function
dims(inds) = ntuple(i -> dim(inds[i]), Val(length(inds)))

# Generic dim function
dim(inds) = prod(dims(inds))

dims(::Tuple{}) = ()

dim(::Tuple{}) = 1

dense(ds::Dims) = ds

dense(::Type{DimsT}) where {DimsT<:Dims} = DimsT

dim(ds::Dims) = prod(ds)

dim(ds::Dims,i::Int) = dims(ds)[i]

mindim(inds::Tuple) = minimum(dims(inds))

mindim(::Tuple{}) = 1

diaglength(inds::Tuple) = mindim(inds)

"""
    strides(ds)

Get the strides from the dimensions.

This is unexported, call with NDTensors.strides.
"""
strides(ds) = Base.size_to_strides(1, dims(ds)...)

"""
    stride(ds, k::Int)

Get the stride of the dimension k from the dimensions.

This is unexported, call with NDTensors.stride.
"""
stride(ds, k::Int) = strides(ds)[k]

# This is to help with some generic programming in the Tensor
# code (it helps to construct a Tuple(::NTuple{N,Int}) where the 
# only known thing for dispatch is a concrete type such
# as Dims{4})
similar_type(::Type{<:Dims},
             ::Type{Val{N}}) where {N} = Dims{N}

# This is to help with ITensor compatibility
dim(i::Int) = i

# This is to help with ITensor compatibility
dir(::Int) = 0

# This is to help with ITensor compatibility
dag(i::Int) = i

# This is to help with ITensor compatibility
sim(i::Int) = i

