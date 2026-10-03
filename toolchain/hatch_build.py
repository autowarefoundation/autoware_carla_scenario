"""Bundles a pinned Codon release into the wheel.

The download happens when the wheel is *built* (in CI, or when uv builds this
workspace member), never when it is installed: installing the wheel only
unpacks files. The archive and license are verified against pinned SHA-256
digests.

Only the parts needed to compile and run programs are kept (bin/ and
lib/codon/: compiler, runtime, standard library); LLVM headers and the
Python JIT bindings are dropped.
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

CODON_VERSION = "0.19.3"
PLATFORMS = {
    # platform key: (archive name, sha256, wheel platform tag)
    "linux-x86_64": (
        "codon-linux-x86_64.tar.gz",
        "f27abda792c0c9f9a42d529c2e0a1b2113ec6964eec23d6f62acef9ed42c3de6",
        # The Codon binaries require glibc >= 2.27.
        "manylinux_2_28_x86_64",
    ),
}
RELEASE_URL = "https://github.com/exaloop/codon/releases/download/v{version}/{archive}"
LICENSE_URL = "https://raw.githubusercontent.com/exaloop/codon/v{version}/LICENSE"
LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
KEEP = ("bin/", "lib/codon/")
LLVM_LICENSE = "include/llvm/Support/LICENSE.TXT"


def _fetch(url: str, sha256: str, cache: Path) -> bytes:
    cached = cache / hashlib.sha256(url.encode()).hexdigest()
    if cached.is_file():
        data = cached.read_bytes()
        if hashlib.sha256(data).hexdigest() == sha256:
            return data
        cached.unlink()  # stale or truncated: download again
    with urllib.request.urlopen(url) as response:  # noqa: S310 (pinned https URL)
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        raise RuntimeError(f"{url}: sha256 {digest} does not match pinned {sha256}")
    cache.mkdir(parents=True, exist_ok=True)
    partial = cached.with_suffix(".partial")
    partial.write_bytes(data)
    os.replace(partial, cached)  # atomic: no truncated cache after an interrupt
    return data


class CodonBundleHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:
        if self.target_name != "wheel":
            return
        platform = os.environ.get("TYPESAFE_CARLA_TOOLCHAIN_PLATFORM", "linux-x86_64")
        if platform not in PLATFORMS:
            raise RuntimeError(f"unsupported platform {platform!r}; known: {sorted(PLATFORMS)}")
        archive, sha256, tag = PLATFORMS[platform]

        root = Path(self.root)
        dest = root / "src" / "typesafe_carla_toolchain" / "codon"
        stamp = dest / ".codon-version"
        want = f"{CODON_VERSION} {platform}\n"
        if not (stamp.is_file() and stamp.read_text() == want):
            cache = Path(os.environ.get("TYPESAFE_CARLA_TOOLCHAIN_CACHE",
                                        root / ".cache" / "downloads"))
            data = _fetch(RELEASE_URL.format(version=CODON_VERSION, archive=archive), sha256, cache)
            license_text = _fetch(LICENSE_URL.format(version=CODON_VERSION), LICENSE_SHA256, cache)
            shutil.rmtree(dest, ignore_errors=True)
            dest.mkdir(parents=True)
            self._extract(data, dest)
            (dest / "LICENSE").write_bytes(license_text)
            stamp.write_text(want)

        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-{tag}"

    @staticmethod
    def _extract(data: bytes, dest: Path) -> None:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for member in tar.getmembers():
                # Strip the top-level codon-deploy-<platform>/ directory.
                parts = member.name.split("/", 1)
                if len(parts) != 2 or not parts[1]:
                    continue
                rel = parts[1]
                if rel == LLVM_LICENSE:
                    target = dest / "LICENSE.LLVM"
                elif rel.startswith(KEEP):
                    target = dest / rel
                else:
                    continue
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as src:
                        target.write_bytes(src.read())
                    target.chmod(member.mode & 0o755 | 0o644)
                elif member.issym():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    # Symlinks do not survive wheels; materialise the target.
                    link = (Path(member.name).parent / member.linkname).as_posix()
                    with tar.extractfile(tar.getmember(link)) as src:
                        target.write_bytes(src.read())
