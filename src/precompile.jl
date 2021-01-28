function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(_gemm!),Char,Char,Float64,ReshapedArray{Float64, 2, Vector{Float64}, Tuple{}},ReshapedArray{Float64, 2, Vector{Float64}, Tuple{}},Float64,ReshapedArray{_A, 2, _B, Tuple{}} where _B<:AbstractArray where _A})   # time: 0.36487946
    Base.precompile(Tuple{typeof(_contract_scalar_perm!),ReshapedArray{_A, _B, _C, Tuple{}} where _C<:AbstractArray where _B where _A,ReshapedArray{Float64, _A, Vector{Float64}, Tuple{}} where _A,Any,Float64,Int64})   # time: 0.2922001
    Base.precompile(Tuple{typeof(contract_labels),Tuple{Int64, Int64},Tuple{Int64, Int64}})   # time: 0.21364069
    Base.precompile(Tuple{typeof(_gemm!),Char,Char,Any,Vector{Float64},Vector{Float64},Any,Tensor{_A, 2, _B, Tuple{Int64, Int64}} where _B<:TensorStorage where _A})   # time: 0.11068376
    Base.precompile(Tuple{typeof(_gemm!),Char,Char,Any,ReshapedArray{Float64, 2, Vector{Float64}, Tuple{}},ReshapedArray{Float64, 2, Vector{Float64}, Tuple{}},Any,ReshapedArray{_A, 2, _B, Tuple{}} where _B<:AbstractArray where _A})   # time: 0.038332146
    Base.precompile(Tuple{typeof(_contract_scalar_perm!),ReshapedArray{Float64, _A, Vector{Float64}, Tuple{}} where _A,ReshapedArray{Float64, _A, Vector{Float64}, Tuple{}} where _A,Any,Float64,Int64})   # time: 0.03757731
    Base.precompile(Tuple{typeof(_contract_scalar!),DenseTensor{ElR, NR, IndsT, StoreT} where StoreT<:Dense where IndsT where NR where ElR<:Number,Tuple{Vararg{T, NR}} where T where NR,Float64,Tuple{Int64, Int64},Float64,Tuple{Int64, Int64},Int64,Int64})   # time: 0.027694328
    Base.precompile(Tuple{typeof(_gemm!),Char,Char,Float64,ReshapedArray{Float64, 2, Vector{Float64}, Tuple{}},ReshapedArray{Float64, 2, Vector{Float64}, Tuple{}},Float64,ReshapedArray{_A, 2, _B, Tuple{}} where _B<:AbstractArray where _A<:Union{Float32, Float64, ComplexF32, ComplexF64}})   # time: 0.025904719
    Base.precompile(Tuple{typeof(contract_labels),Type{Val{2}},Tuple{Int64, Int64},Tuple{Int64, Int64}})   # time: 0.020031951
    Base.precompile(Tuple{typeof(_gemm!),Char,Char,Any,Vector{Float64},Vector{Float64},Any,Tensor{_A, 2, _B, Tuple{Int64, Int64}} where _B<:TensorStorage where _A<:Union{Float32, Float64, ComplexF32, ComplexF64}})   # time: 0.017152105
    Base.precompile(Tuple{typeof(_contract_scalar_perm!),ReshapedArray{_A, _B, _C, Tuple{}} where _C<:AbstractArray where _B where _A,ReshapedArray{Float64, _A, Vector{Float64}, Tuple{}} where _A,Any,Any,Any})   # time: 0.015323097
    Base.precompile(Tuple{typeof(copy),DenseTensor{ElR, NR, IndsT, StoreT} where StoreT<:Dense where IndsT where NR where ElR<:Number})   # time: 0.014552362
    Base.precompile(Tuple{typeof(compute_perms!),ContractionProperties{2, 2, _A} where _A})   # time: 0.012716902
    Base.precompile(Tuple{typeof(compute_perms!),ContractionProperties{2, 2, 2}})   # time: 0.006823508
    Base.precompile(Tuple{typeof(_contract_scalar!),DenseTensor{ElR, 2, IndsT, StoreT} where StoreT<:Dense where IndsT where ElR<:Number,Tuple{Int64, Int64},Float64,Tuple{Int64, Int64},Float64,Tuple{Int64, Int64},Int64,Int64})   # time: 0.006470757
    Base.precompile(Tuple{typeof(_gemm!),Char,Char,Float64,ReshapedArray{Float64, 2, Vector{Float64}, Tuple{}},ReshapedArray{Float64, 2, Vector{Float64}, Tuple{}},Float64,ReshapedArray{Float64, 2, Matrix{Float64}, Tuple{}}})   # time: 0.006365003
    Base.precompile(Tuple{typeof(_contract_scalar!),DenseTensor{ElR, NR, IndsT, StoreT} where StoreT<:Dense where IndsT where NR where ElR<:Number,Tuple{Vararg{T, NR}} where T where NR,Float64,Tuple{Int64, Int64},Float64,Tuple{Int64, Int64},Any,Any})   # time: 0.005038261
    #Base.precompile(Tuple{Type{ContractionProperties},Tuple{Int64, Int64},Tuple{Int64, Int64},Core.Tuple{Core.Vararg{Core.Int64, NC}}})   # time: 0.004911812
    Base.precompile(Tuple{typeof(similar),Type{DenseTensor{Float64, _A, _B, Dense{Float64, Vector{Float64}}}} where _B where _A,Tuple{Vararg{Int64, N}} where N})   # time: 0.004319293
    Base.precompile(Tuple{typeof(_contract_scalar_perm!),ReshapedArray{Float64, _A, Vector{Float64}, Tuple{}} where _A,ReshapedArray{Float64, _A, Vector{Float64}, Tuple{}} where _A,Any,Float64,Float64})   # time: 0.004009066
    Base.precompile(Tuple{typeof(reshape),DenseTensor{ElR, NR, IndsT, StoreT} where StoreT<:Dense where IndsT where NR where ElR<:Number,Int64,Int64})   # time: 0.002678546
    Base.precompile(Tuple{typeof(nnz),DenseTensor{ElT, N, IndsT, StoreT} where StoreT<:Dense where IndsT where N where ElT})   # time: 0.00263423
    Base.precompile(Tuple{typeof(_gemm!),Char,Char,Float64,ReshapedArray{Float64, 2, Vector{Float64}, Tuple{}},ReshapedArray{Float64, 2, Vector{Float64}, Tuple{}},Float64,ReshapedArray{Float64, 2, Vector{Float64}, Tuple{}}})   # time: 0.002312892
    Base.precompile(Tuple{typeof(_gemm!),Char,Char,Any,ReshapedArray{Float64, 2, Vector{Float64}, Tuple{}},ReshapedArray{Float64, 2, Vector{Float64}, Tuple{}},Any,ReshapedArray{_A, 2, _B, Tuple{}} where _B<:AbstractArray where _A<:Union{Float32, Float64, ComplexF32, ComplexF64}})   # time: 0.002187426
    Base.precompile(Tuple{typeof(checkBCsameord),ContractionProperties{2, 2, _A} where _A})   # time: 0.001272044
    Base.precompile(Tuple{typeof(checkACsameord),ContractionProperties{2, 2, _A} where _A})   # time: 0.001178859
end
