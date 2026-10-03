# typesafe-carla-toolchain

The Codon compiler, pinned and repackaged as a wheel for
[typesafe-carla](https://github.com/hakuturu583/typesafe_carla). Installing it
needs no network access or post-install step; `typesafe-codon` finds it
automatically.

| Package version | Codon | Platform |
|---|---|---|
| 0.19.3 | [v0.19.3](https://github.com/exaloop/codon/releases/tag/v0.19.3) | Linux x86_64 (glibc ≥ 2.28) |

`codon build` links executables with the system C++ compiler (`g++`) and
zlib (`-lz`), so they must be installed to produce executables; `codon run`
does not need them.

Codon is licensed under the Apache License 2.0 (bundled as
`typesafe_carla_toolchain/codon/LICENSE`); its LLVM components under the
Apache License 2.0 with LLVM exceptions (`codon/LICENSE.LLVM`). The bundled
GCC runtime libraries (`libgcc_s`, `libgfortran`, `libquadmath`) are covered
by the GCC Runtime Library Exception.
