#!/usr/bin/env python3
"""
verify_env.py — Pre-flight environment verification for meeting-summarizer.

Checks:
  1. ARM64 native Python (no Rosetta x86 emulation)
  2. PyTorch importable with correct version
  3. PyTorch built with MPS support
  4. MPS device available at runtime (requires Xcode toolchain)
  5. BF16 tensor creation on MPS succeeds (M4 Pro native)
  6. Float64 on MPS is correctly rejected (prevents silent CPU fallback)

Exits with code 0 on full pass, code 1 on any failure.
"""

import platform
import sys

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "


def _run_check(label: str, fn) -> bool:
    """Execute a single check function; print formatted pass/fail. Returns True on pass."""
    try:
        detail = fn()
        suffix = f"  ({detail})" if isinstance(detail, str) else ""
        print(f"  {PASS}  {label}{suffix}")
        return True
    except AssertionError as exc:
        print(f"  {FAIL}  {label}")
        print(f"       → {exc}")
        return False
    except Exception as exc:
        print(f"  {FAIL}  {label}")
        print(f"       → {type(exc).__name__}: {exc}")
        return False


def main() -> None:
    print()
    print("=" * 62)
    print("  meeting-summarizer · Environment Verification")
    print("=" * 62)
    print()

    failures: list[str] = []

    # ── Check 1: ARM64 native ─────────────────────────────────────────────
    def chk_arm64() -> str:
        machine = platform.machine()
        assert machine == "arm64", (
            f"Expected arm64, got '{machine}' — Python is running under Rosetta. "
            "Reinstall Python 3.12 from Homebrew on ARM64."
        )
        return f"platform.machine() = {machine}"

    if not _run_check("ARM64 native Python (no Rosetta)", chk_arm64):
        failures.append("ARM64 check")

    # ── Import torch (prerequisite for remaining checks) ─────────────────
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        print(f"  {FAIL}  PyTorch import — not installed")
        print(f"       → Run: pip install torch torchvision torchaudio")
        print()
        print("=" * 62)
        print(f"  {FAIL}  ENVIRONMENT NOT READY — torch missing, cannot continue")
        print("=" * 62)
        sys.exit(1)

    # ── Check 2: PyTorch version readable ────────────────────────────────
    def chk_torch_version() -> str:
        return f"torch {torch.__version__}"

    if not _run_check("PyTorch importable", chk_torch_version):
        failures.append("PyTorch import")

    # ── Check 3: MPS compiled into this wheel ────────────────────────────
    def chk_mps_built() -> str:
        assert torch.backends.mps.is_built(), (
            "PyTorch was not built with MPS support. "
            "Reinstall: pip install torch --force-reinstall"
        )
        return "torch.backends.mps.is_built() = True"

    if not _run_check("PyTorch built with MPS support", chk_mps_built):
        failures.append("MPS built-in")

    # ── Check 4: MPS available at runtime ────────────────────────────────
    def chk_mps_available() -> str:
        assert torch.backends.mps.is_available(), (
            "MPS not available at runtime. "
            "Re-run D0.2: sudo xcode-select -s /Users/vnissankararao/Downloads/Xcode.app/Contents/Developer"
        )
        return "torch.backends.mps.is_available() = True"

    if not _run_check("MPS device available at runtime", chk_mps_available):
        failures.append("MPS available")

    # ── Check 5: BF16 tensor on MPS ──────────────────────────────────────
    def chk_bf16() -> str:
        t = torch.zeros(4, dtype=torch.bfloat16, device="mps")
        assert t.dtype == torch.bfloat16, f"Expected bfloat16, got {t.dtype}"
        assert str(t.device).startswith("mps"), f"Expected mps device, got {t.device}"
        del t
        torch.mps.empty_cache()
        return "dtype=bfloat16 on mps — M4 Pro native BF16 confirmed"

    if not _run_check("BF16 tensor creation on MPS", chk_bf16):
        failures.append("BF16 on MPS")

    # ── Check 6: Float64 rejected by MPS ─────────────────────────────────
    print(f"  ", end="")
    try:
        t64 = torch.zeros(4, dtype=torch.float64, device="mps")
        del t64
        # Not a hard failure — but warn clearly
        print(f"{WARN} Float64 on MPS did not raise — watch for silent CPU fallback")
    except (RuntimeError, TypeError) as exc:
        print(f"{PASS}  Float64 rejected by MPS  ({type(exc).__name__})")

    # ── Memory baseline ───────────────────────────────────────────────────
    print()
    try:
        driver_mb = torch.mps.driver_allocated_memory() / 1e6
        print(f"  MPS driver memory baseline : {driver_mb:.1f} MB")
    except Exception as exc:
        print(f"  {WARN} Could not read MPS memory baseline: {exc}")

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    if failures:
        print(f"  {FAIL}  ENVIRONMENT NOT READY — {len(failures)} failure(s):")
        for item in failures:
            print(f"       • {item}")
        print("=" * 62)
        print()
        sys.exit(1)
    else:
        print(f"  {PASS}  ALL CHECKS PASSED — environment is ready for training")
        print("=" * 62)
        print()
        sys.exit(0)


if __name__ == "__main__":
    main()
