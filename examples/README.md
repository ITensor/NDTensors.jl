This is an example of using TBLIS as a contraction backend.

First install TBLIS.jl: https://github.com/mtfishman/TBLIS.jl

TBLIS is enabled by the command `using TBLIS`. Once it is enabled, it can be
disabled with the command `disable_tblis()`, and enabled again with the command
`enable_tblis()`. You can then set the number of threads with `TBLIS.set_num_threads(n)`.

Currently only tensor contractions involving real elements (Float32 and Float64)
get dispatched to TBLIS, since the complex number support of TBLIS is limited (https://github.com/devinamatthews/tblis/issues/18).

