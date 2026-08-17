import os
import re
import subprocess
from pathlib import Path


def _prefer_pip_cuda_home() -> None:
    """Point CUDA_HOME at pip nvidia-cuda-nvcc when unset (match torch's CUDA)."""
    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        return
    try:
        from importlib.metadata import PackageNotFoundError, distribution  # noqa: PLC0415

        root = Path(str(distribution("nvidia-cuda-nvcc").locate_file("")))
    except PackageNotFoundError:
        return
    for candidate in (root / "nvidia" / "cu13", root / "nvidia" / "cuda_nvcc"):
        if (candidate / "bin" / "nvcc").is_file():
            os.environ["CUDA_HOME"] = str(candidate)
            path = os.environ.get("PATH", "")
            bin_dir = str(candidate / "bin")
            if bin_dir not in path.split(os.pathsep):
                os.environ["PATH"] = bin_dir + os.pathsep + path
            return


_prefer_pip_cuda_home()

import setuptools  # CUDA_HOME must be set before cpp_extension import
import torch
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent


def _cuda_lib_dirs() -> list[Path]:
    """Real lib dirs under CUDA_HOME (pip wheels use ``lib/``, system toolkits ``lib64/``)."""
    if CUDA_HOME is None:
        return []
    home = Path(CUDA_HOME)
    return [p for p in (home / "lib", home / "lib64", home / "lib64" / "stubs", home / "lib" / "stubs") if p.is_dir()]


def _system_cuda_stub_dirs() -> list[Path]:
    """Dirs that provide ``libcuda.so`` (driver stub). Pip CUDA wheels do not ship it."""
    candidates = [
        Path("/usr/local/cuda/lib64/stubs"),
        Path("/usr/local/cuda/lib/stubs"),
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/usr/lib64"),
    ]
    # If CUDA_HOME is the pip tree, also probe a sibling system toolkit.
    if CUDA_HOME is not None:
        home = Path(CUDA_HOME)
        candidates.extend([home / "lib64" / "stubs", home / "lib" / "stubs"])
    return [p for p in candidates if (p / "libcuda.so").is_file() or (p / "libcuda.so.1").is_file()]


def _pip_cuda_link_dirs() -> list[str]:
    """Library dirs for ``-lcudart`` / ``-lnvrtc`` / ``-lcuda`` against pip CUDA wheels.
    Pip nvidia wheels ship versioned sonames only (``libcudart.so.13``) without the
    unversioned ``libcudart.so`` that ``-lcudart`` expects. Rather than mutate
    site-packages, stage relative symlinks in a build-local dir and put that first
    on the linker search path. System toolkits already have unversioned names, so
    the staging dir stays empty of conflicts and later dirs still work.
    ``libcuda`` (driver) is never in the pip wheels; append system stub dirs for
    ``-lcuda`` (blockwise).
    """
    real_dirs = _cuda_lib_dirs()
    stub_dirs = _system_cuda_stub_dirs()
    if not real_dirs and not stub_dirs:
        return []

    stage = ROOT / "build" / "cuda_so_links"
    stage.mkdir(parents=True, exist_ok=True)

    # Prefer the longest soname (libfoo.so.13.2.75 over libfoo.so.13) when several exist.
    by_stem: dict[str, Path] = {}
    for lib_dir in real_dirs:
        for versioned in lib_dir.glob("lib*.so.*"):
            if versioned.name.endswith(".a"):
                continue
            stem = versioned.name.split(".so.", 1)[0] + ".so"
            prev = by_stem.get(stem)
            if prev is None or len(versioned.name) >= len(prev.name):
                by_stem[stem] = versioned

    for stem, versioned in by_stem.items():
        link = stage / stem
        if link.is_symlink() or link.exists():
            if link.is_symlink() and link.resolve() == versioned.resolve():
                continue
            link.unlink()
        # Relative target so the stage dir is relocatable within the build tree.
        link.symlink_to(os.path.relpath(versioned, start=stage))

    # Dedupe while preserving order: stage, pip libs, system stubs.
    out: list[str] = [str(stage)]
    seen = {str(stage)}
    for p in [*real_dirs, *stub_dirs]:
        s = str(p)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# cutlass headers for the blockwise GEMM kernels. Pinned to the upstream commit the
# build was validated against and fetched into a cache dir, rather than carried as a
# git submodule (keeps the public-repo sync clean and the clone CI-cacheable).
CUTLASS_REPO = "https://github.com/NVIDIA/cutlass.git"
CUTLASS_REF = "afa1772203677c5118fcd82537a9c8fefbcc7008"  # v3.8.0


def _nvidia_include_dirs() -> list[str]:
    """Include dirs from pip-installed nvidia packages (e.g. cusparse headers)."""
    try:
        import nvidia  # noqa: PLC0415

        return [str(p) for pkg in Path(nvidia.__path__[0]).iterdir() if (p := pkg / "include").is_dir()]
    except ImportError:
        return []


def _arch_tokens() -> list[str]:
    """Normalized entries from TORCH_CUDA_ARCH_LIST (e.g. ['8.9', '9.0'])."""
    raw = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    return [re.sub(r"\+PTX$", "", tok).strip() for tok in raw.replace(",", " ").split() if tok.strip()]


# Arch codes the blockwise FP8 GEMM kernels support, as nvcc `sm_<code>` targets.
# SM89 ("geforce") is the generic fp8 kernel: it runs on Ada and is also the kernel
# dispatched on Blackwell (sm_100a datacenter / sm_120 consumer), so it is compiled for
# those too. The SM90 ("deep_gemm") kernel is Hopper-only and needs sm_90a (wgmma/TMA);
# it is declared-always / defined-conditionally, so it stubs out on every non-Hopper
# pass and can share a multi-arch fat binary. Ampere has no fp8 path. Entries are
# filtered to what the local nvcc can actually target (see _nvcc_arch_nums), so the
# Blackwell codes are inert until built with CUDA 12.8+.
# NOTE: the Blackwell (100a/120) path is implemented but not yet validated on real
# Blackwell hardware -- needs a B200 + CUDA 12.8 build/run pass.
_BLOCKWISE_ARCHES = ["89", "90a", "100a", "120"]


def _nvcc_arch_nums() -> set[str]:
    """Architecture numbers this nvcc can target, e.g. {'80', '86', '89', '90'}."""
    try:
        out = subprocess.check_output([f"{CUDA_HOME}/bin/nvcc", "--list-gpu-arch"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return set()
    return {m.group(1) for tok in out.split() if (m := re.match(r"compute_(\d+a?)$", tok.strip()))}


def _blockwise_gencode() -> tuple[list[str], bool]:
    """Return (``-gencode`` flags for blockwise_cpp, build_sm90).
    Honors ``TORCH_CUDA_ARCH_LIST`` when set (mapping 8.9 -> sm_89, 9.0/9.0a -> sm_90a,
    ignoring arches the kernels do not support); unset builds for every supported arch
    this nvcc can target. The flags apply uniformly to all sources -- safe because the
    SM90 source stubs itself out on non-sm_90a passes.
    """
    supported = _nvcc_arch_nums()
    # nvcc may report an arch either plain ("90") or suffixed ("90a"); accept either.
    base = [a for a in _BLOCKWISE_ARCHES if a in supported or a.rstrip("a") in supported]
    env = _arch_tokens()
    if env:
        sel = []
        for tok in env:
            if tok.startswith("8.9"):
                sel.append("89")
            elif tok.startswith("9.0"):
                sel.append("90a")
            elif tok.startswith("10.0"):
                sel.append("100a")
            elif tok.startswith("12.0"):
                sel.append("120")
            # Ampere (8.0/8.6) and other arches have no fp8 blockwise kernel.
        archs = [a for a in dict.fromkeys(sel) if a in base] or base
    else:
        archs = base
    flags = [f"-gencode=arch=compute_{a},code=sm_{a}" for a in archs]
    return flags, ("90a" in archs)


# Arch codes the NVFP4 kernels support. The E2M1 pack/convert intrinsics and the cuBLASLt
# block-scaled FP4 kernels are Blackwell-only (sm_100a datacenter, sm_120a consumer).
_NVFP4_ARCHES = ["100a", "120a"]


def _nvfp4_gencode() -> list[str]:
    """Blackwell arches to build the NVFP4 extension for, or [] to skip it."""
    supported = _nvcc_arch_nums()
    base = [a for a in _NVFP4_ARCHES if a in supported or a.rstrip("a") in supported]
    env = _arch_tokens()
    if not env:
        return base
    sel = []
    for tok in env:
        if tok.startswith("10.0"):
            sel.append("100a")
        elif tok.startswith("12.0"):
            sel.append("120a")
    return [a for a in dict.fromkeys(sel) if a in base]


def _cutlass_include() -> str:
    """Return the cutlass include dir, fetching the pinned commit on first use.
    Honors ``CUTLASS_DIR`` (a prebuilt cutlass checkout, e.g. a system copy) and
    otherwise caches a shallow clone of ``CUTLASS_REF`` under
    ``LTX_KERNELS_CACHE_DIR`` (default ``~/.cache/ltx-kernels``), so it is reused
    across builds and can be restored from a CI cache. Keeps ``uv sync`` /
    ``pip install -e`` self-contained without a git submodule.
    """
    if env := os.environ.get("CUTLASS_DIR"):
        return str(Path(env) / "include")
    cache_root = Path(os.environ.get("LTX_KERNELS_CACHE_DIR", Path.home() / ".cache" / "ltx-kernels"))
    dest = cache_root / f"cutlass-{CUTLASS_REF}"
    if not (dest / "include" / "cutlass" / "cutlass.h").is_file():
        dest.mkdir(parents=True, exist_ok=True)
        # Blobless partial clone of the exact pinned commit (GitHub allows fetching an
        # arbitrary SHA), sparse-checked-out to include/ only: cutlass is header-only and
        # the rest of the repo (tools/test/examples/python, ~85% by size) is unused.
        subprocess.run(["git", "init", "-q", str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "remote", "add", "origin", CUTLASS_REPO], check=True)
        # Cone-mode sparse checkout of just include/. Use init + set (not "set --cone",
        # whose inline flag postdates git 2.35 and is silently parsed as a pattern on older git).
        subprocess.run(["git", "-C", str(dest), "sparse-checkout", "init", "--cone"], check=True)
        subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", "include"], check=True)
        subprocess.run(
            ["git", "-C", str(dest), "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", CUTLASS_REF],
            check=True,
        )
        subprocess.run(["git", "-C", str(dest), "checkout", "-q", "FETCH_HEAD"], check=True)
    return str(dest / "include")


if __name__ == "__main__":
    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA toolkit not found (CUDA_HOME is None). ltx-kernels compiles CUDA extensions "
            "and must be built on a host with the CUDA toolkit installed (nvcc on PATH or "
            "CUDA_HOME set)."
        )

    cuda_link_dirs = _pip_cuda_link_dirs()
    ext_modules = []

    # all2all_cpp -- unchanged.
    all2all_args = [
        "-O3",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-Wno-unused-parameter",
        "-Wno-attributes",
        # CUDA 12.8's cuSPARSE headers declare legacy API types as deprecated.
        # Those third-party declarations are pulled in transitively by PyTorch
        # and must not fail this extension's otherwise strict warning policy.
        "-Wno-deprecated-declarations",
    ]
    ext_modules.append(
        CUDAExtension(
            name="all2all_cpp",
            include_dirs=[str(ROOT / "csrc/all2all"), str(ROOT / "csrc/include"), *_nvidia_include_dirs()],
            sources=[
                "csrc/all2all/all2all.cpp",
                "csrc/all2all/cuda/all2all_heads.cu",
                "csrc/all2all/cuda/allgather.cu",
            ],
            library_dirs=cuda_link_dirs,
            extra_compile_args={"cxx": all2all_args, "nvcc": ["-O3"]},
        )
    )

    # ops_cpp -- arch-independent element ops (rms_norm_rope, rms_norm_split_rope,
    # fp6 pack/unpack). Arch is driven by TORCH_CUDA_ARCH_LIST / torch defaults.
    ext_modules.append(
        CUDAExtension(
            name="ops_cpp",
            sources=[
                "csrc/ops/ops_api.cpp",
                "csrc/ops/fp6_bitpack.cpp",
                "csrc/ops/fp6_pack.cu",
                "csrc/ops/rms_norm_rope.cpp",
                "csrc/ops/rms_norm_rope_cuda.cu",
                "csrc/ops/rms_norm_split_rope.cpp",
                "csrc/ops/rms_norm_split_rope_cuda.cu",
            ],
            include_dirs=[str(ROOT / "csrc/ops/include"), *_nvidia_include_dirs()],
            library_dirs=cuda_link_dirs,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",
                ],
            },
        )
    )

    # blockwise_cpp -- FP8 GEMM. The SM89 (GeForce) kernel is always built; the SM90
    # (deep_gemm) kernel + -D__SM90__ are added whenever sm_90a is among the targets.
    # Arches are an explicit -gencode list (see _blockwise_gencode): TORCH_CUDA_ARCH_LIST
    # when set, else every supported arch this nvcc can target ("build for everything").
    # The list is uniform across sources -- the SM90 source declares-always /
    # defines-conditionally, so it compiles (as a stub) for non-sm_90a arches too. Note
    # blockwise is unsupported on Ampere and fails at *runtime* there, by design.
    cutlass_include = _cutlass_include()
    gencode, build_sm90 = _blockwise_gencode()
    blockwise_sources = [
        "csrc/blockwise/api.cpp",
        "csrc/blockwise/kernels/geforce/gemm.cu",
    ]
    abi = f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}"
    blockwise_cxx = ["-O3", "-std=c++17", "-fPIC", "-Wno-psabi", "-Wno-deprecated-declarations", abi]
    blockwise_nvcc = [
        "-O3",
        "-std=c++17",
        "--ptxas-options=-O2",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",
        *gencode,
    ]
    if build_sm90:
        blockwise_sources.append("csrc/blockwise/kernels/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d_bias.cu")
        blockwise_cxx.append("-D__SM90__")
        blockwise_nvcc.append("-D__SM90__")

    ext_modules.append(
        CUDAExtension(
            name="blockwise_cpp",
            sources=blockwise_sources,
            include_dirs=[
                f"{CUDA_HOME}/include",
                f"{CUDA_HOME}/include/cccl",
                str(ROOT / "csrc/blockwise"),
                str(ROOT / "csrc/blockwise/kernels/deep_gemm/include"),
                cutlass_include,
                *_nvidia_include_dirs(),
            ],
            libraries=["cuda", "cudart", "nvrtc"],
            library_dirs=cuda_link_dirs,
            extra_compile_args={"cxx": blockwise_cxx, "nvcc": blockwise_nvcc},
        )
    )

    # nvfp4_cpp -- in-house NVFP4 quantize (CUDA) + block-scaled GEMM (cuBLASLt).
    # Blackwell-only: the E2M1/E4M3 conversion intrinsics and the cuBLASLt FP4 kernels
    # both need SM >= 10.0, so this extension is built for the Blackwell targets among
    # TORCH_CUDA_ARCH_LIST (defaulting to whatever this nvcc supports) and is skipped
    # entirely when nvcc is too old to emit them.
    nvfp4_archs = _nvfp4_gencode()
    if nvfp4_archs:
        ext_modules.append(
            CUDAExtension(
                name="nvfp4_cpp",
                sources=[
                    "csrc/nvfp4/api.cpp",
                    "csrc/nvfp4/gemm.cpp",
                    "csrc/nvfp4/quantize.cu",
                ],
                include_dirs=[
                    str(ROOT / "csrc/nvfp4"),
                    f"{CUDA_HOME}/include",
                    *_nvidia_include_dirs(),
                ],
                libraries=["cublasLt"],
                library_dirs=cuda_link_dirs,
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17"],
                    "nvcc": [
                        "-O3",
                        "-std=c++17",
                        "--expt-relaxed-constexpr",
                        "--expt-extended-lambda",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                        "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",
                        *[f"-gencode=arch=compute_{a},code=sm_{a}" for a in nvfp4_archs],
                    ],
                },
            )
        )

    setuptools.setup(
        ext_modules=ext_modules,
        cmdclass={"build_ext": BuildExtension},
    )
