# NDTensors.jl


| **Build Status**                                                                                |
|:-----------------------------------------------------------------------------------------------:|
| [![Tests](https://github.com/ITensor/NDTensors.jl/workflows/Tests/badge.svg)](https://github.com/ITensor/NDTensors.jl/actions?query=workflow%3ATests) [![codecov](https://codecov.io/gh/ITensor/NDTensors.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/ITensor/NDTensors.jl) |

THIS PACKAGE IS A PRERELEASE AND IS SUBJECT TO BREAKING CHANGES. FOR NOW IT IS MOSTLY MEANT FOR USE WITHIN THE ITENSORS.JL PACKAGE.

Please use within your own packages with caution, as minor releases may be breaking and we may not be careful about deprecations between v0.1 and v0.2 (for example, code may break if you upgrade from v0.1.1 to v0.1.2). Our plan is for versions after v0.2 to be more stable.

NDTensors is a Julia package for n-dimensional sparse tensors. For now, it supports dense, block sparse, diagonal, and diagonal block sparse tensors. The focus is on providing efficient tensor operations, such as tensor decompositions and contractions.
