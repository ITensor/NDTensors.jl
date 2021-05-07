# NDTensors.jl


| **Build Status**                                                                                |
|:-----------------------------------------------------------------------------------------------:|
| [![Tests](https://github.com/ITensor/NDTensors.jl/workflows/Tests/badge.svg)](https://github.com/ITensor/NDTensors.jl/actions?query=workflow%3ATests) [![codecov](https://codecov.io/gh/ITensor/NDTensors.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/ITensor/NDTensors.jl) |

**NOTE: For the time being, development of the NDTensors module will happen within [ITensors.jl](https://github.com/ITensor/ITensors.jl).** This repository and package will remain as-is but will not be developed. If you plan to contribute to the NDTensors functionality, please do so in the `ITensors.jl` repository (the `NDTensors` module has been moved into the `src/NDTensors` folder in that repository). Additionally, if you would like to use the latest version of `NDTensors`, you can do so by installing `ITensors.jl` with `import Pkg; Pkg.add("ITensors")` and use the `NDTensors` module with `using ITensors.NDTensors`. Additionally, if your package depends directly on `NDTensors`, you should change the dependency to just depend on `ITensors`. This change is meant to ease the development of NDTensors in conjunction with ITensors and simplify the testing and benchmarking of NDTensors and ITensors. Once `NDTensors` is more stable and tested, we plan to move development back to this repository.

`NDTensors` is a Julia package for n-dimensional sparse tensors. For now, it supports dense, block sparse, diagonal, and diagonal block sparse tensors. The focus is on providing efficient tensor operations, such as tensor decompositions and contractions.
