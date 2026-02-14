"""
EdgeTutor AI — Jetson hardware detection and adaptive model selection.

Detects which Jetson board is running, how much RAM/VRAM is available,
and recommends appropriate model configurations. Provides automatic
fallback to smaller models if the system can't handle the default.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from edgetutor.core.logging_config import get_logger

logger = get_logger(__name__)


# ── Known Jetson boards and their specs ───────────────────────────────────────
@dataclass
class JetsonProfile:
    """Hardware profile for a Jetson board."""

    name: str
    ram_gb: float
    gpu_name: str
    gpu_ram_shared: bool  # Jetson uses unified memory
    cuda_cores: int
    recommended_llm: str  # GGUF model filename/identifier
    recommended_llm_url: str
    recommended_llm_size_gb: float
    recommended_gpu_layers: int
    recommended_context_size: int
    recommended_stt_model: str
    recommended_stt_compute: str
    max_llm_params_b: float  # Max parameter count in billions
    notes: str


JETSON_PROFILES = {
    "orin_nano_8gb": JetsonProfile(
        name="Jetson Orin Nano 8GB",
        ram_gb=8.0,
        gpu_name="Ampere (1024 CUDA cores)",
        gpu_ram_shared=True,
        cuda_cores=1024,
        recommended_llm="Phi-3-mini-4k-instruct-q4.gguf",
        recommended_llm_url="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
        recommended_llm_size_gb=2.3,
        recommended_gpu_layers=20,
        recommended_context_size=2048,
        recommended_stt_model="small",
        recommended_stt_compute="float16",
        max_llm_params_b=7.0,
        notes="Best balance of quality and speed for tutoring.",
    ),
    "orin_nano_4gb": JetsonProfile(
        name="Jetson Orin Nano 4GB",
        ram_gb=4.0,
        gpu_name="Ampere (512 CUDA cores)",
        gpu_ram_shared=True,
        cuda_cores=512,
        recommended_llm="tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
        recommended_llm_url="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
        recommended_llm_size_gb=1.1,
        recommended_gpu_layers=10,
        recommended_context_size=1024,
        recommended_stt_model="tiny",
        recommended_stt_compute="int8",
        max_llm_params_b=3.0,
        notes="Memory-constrained. Use smallest models.",
    ),
    "orin_nx_8gb": JetsonProfile(
        name="Jetson Orin NX 8GB",
        ram_gb=8.0,
        gpu_name="Ampere (1024 CUDA cores)",
        gpu_ram_shared=True,
        cuda_cores=1024,
        recommended_llm="Phi-3-mini-4k-instruct-q4.gguf",
        recommended_llm_url="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
        recommended_llm_size_gb=2.3,
        recommended_gpu_layers=25,
        recommended_context_size=2048,
        recommended_stt_model="small",
        recommended_stt_compute="float16",
        max_llm_params_b=7.0,
        notes="Similar to Orin Nano 8GB but faster GPU.",
    ),
    "orin_nx_16gb": JetsonProfile(
        name="Jetson Orin NX 16GB",
        ram_gb=16.0,
        gpu_name="Ampere (2048 CUDA cores)",
        gpu_ram_shared=True,
        cuda_cores=2048,
        recommended_llm="Phi-3-mini-4k-instruct-q4.gguf",
        recommended_llm_url="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
        recommended_llm_size_gb=2.3,
        recommended_gpu_layers=33,
        recommended_context_size=4096,
        recommended_stt_model="medium",
        recommended_stt_compute="float16",
        max_llm_params_b=13.0,
        notes="Plenty of memory. Can run larger models and context.",
    ),
    "agx_orin_32gb": JetsonProfile(
        name="Jetson AGX Orin 32GB",
        ram_gb=32.0,
        gpu_name="Ampere (2048 CUDA cores)",
        gpu_ram_shared=True,
        cuda_cores=2048,
        recommended_llm="Phi-3-mini-4k-instruct-q4.gguf",
        recommended_llm_url="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
        recommended_llm_size_gb=2.3,
        recommended_gpu_layers=33,
        recommended_context_size=4096,
        recommended_stt_model="medium",
        recommended_stt_compute="float16",
        max_llm_params_b=30.0,
        notes="Can run 7B+ models at higher quality. Consider Q5_K_M or Q6_K.",
    ),
    "agx_orin_64gb": JetsonProfile(
        name="Jetson AGX Orin 64GB",
        ram_gb=64.0,
        gpu_name="Ampere (2048 CUDA cores)",
        gpu_ram_shared=True,
        cuda_cores=2048,
        recommended_llm="Phi-3-mini-4k-instruct-q4.gguf",
        recommended_llm_url="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
        recommended_llm_size_gb=2.3,
        recommended_gpu_layers=33,
        recommended_context_size=4096,
        recommended_stt_model="medium",
        recommended_stt_compute="float16",
        max_llm_params_b=70.0,
        notes="Flagship. Can run 13B+ models easily.",
    ),
    # Older Jetson boards (Xavier, Nano)
    "xavier_nx": JetsonProfile(
        name="Jetson Xavier NX",
        ram_gb=8.0,
        gpu_name="Volta (384 CUDA cores)",
        gpu_ram_shared=True,
        cuda_cores=384,
        recommended_llm="tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
        recommended_llm_url="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
        recommended_llm_size_gb=1.1,
        recommended_gpu_layers=10,
        recommended_context_size=1024,
        recommended_stt_model="tiny",
        recommended_stt_compute="int8",
        max_llm_params_b=3.0,
        notes="Older GPU arch. Use small models. Slower inference.",
    ),
    "nano_4gb": JetsonProfile(
        name="Jetson Nano 4GB (original)",
        ram_gb=4.0,
        gpu_name="Maxwell (128 CUDA cores)",
        gpu_ram_shared=True,
        cuda_cores=128,
        recommended_llm="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        recommended_llm_url="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        recommended_llm_size_gb=0.67,
        recommended_gpu_layers=5,
        recommended_context_size=512,
        recommended_stt_model="tiny",
        recommended_stt_compute="int8",
        max_llm_params_b=1.5,
        notes="Very constrained. Minimal viable experience.",
    ),
    # Generic fallback for non-Jetson or unrecognized
    "generic_gpu": JetsonProfile(
        name="Generic GPU System",
        ram_gb=8.0,
        gpu_name="Unknown",
        gpu_ram_shared=False,
        cuda_cores=0,
        recommended_llm="Phi-3-mini-4k-instruct-q4.gguf",
        recommended_llm_url="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
        recommended_llm_size_gb=2.3,
        recommended_gpu_layers=20,
        recommended_context_size=2048,
        recommended_stt_model="small",
        recommended_stt_compute="float16",
        max_llm_params_b=7.0,
        notes="Non-Jetson system. Defaults may need tuning.",
    ),
    "cpu_only": JetsonProfile(
        name="CPU-only System",
        ram_gb=4.0,
        gpu_name="None",
        gpu_ram_shared=False,
        cuda_cores=0,
        recommended_llm="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        recommended_llm_url="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        recommended_llm_size_gb=0.67,
        recommended_gpu_layers=0,
        recommended_context_size=1024,
        recommended_stt_model="tiny",
        recommended_stt_compute="int8",
        max_llm_params_b=3.0,
        notes="No GPU. LLM runs on CPU only (slow).",
    ),
}

# ── Model tiers for fallback ─────────────────────────────────────────────────
MODEL_TIERS = [
    {
        "name": "Phi-3 Mini Q4 (3.8B)",
        "filename": "Phi-3-mini-4k-instruct-q4.gguf",
        "url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
        "size_gb": 2.3,
        "params_b": 3.8,
        "min_ram_gb": 6.0,
        "quality": "high",
        "context": 4096,
    },
    {
        "name": "TinyLlama 1.1B Q8",
        "filename": "tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
        "size_gb": 1.1,
        "params_b": 1.1,
        "min_ram_gb": 3.0,
        "quality": "medium",
        "context": 2048,
    },
    {
        "name": "TinyLlama 1.1B Q4",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "size_gb": 0.67,
        "params_b": 1.1,
        "min_ram_gb": 2.0,
        "quality": "low",
        "context": 1024,
    },
]


def _read_proc_file(path: str) -> str:
    """Read a /proc file, return empty string on failure."""
    try:
        with open(path) as f:
            return f.read().strip()
    except (FileNotFoundError, PermissionError):
        return ""


def get_total_ram_gb() -> float:
    """Get total system RAM in GB."""
    try:
        meminfo = _read_proc_file("/proc/meminfo")
        for line in meminfo.split("\n"):
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                return round(kb / (1024 * 1024), 1)
    except Exception:
        pass
    return 0.0


def get_available_ram_gb() -> float:
    """Get available system RAM in GB."""
    try:
        meminfo = _read_proc_file("/proc/meminfo")
        for line in meminfo.split("\n"):
            if line.startswith("MemAvailable:"):
                kb = int(line.split()[1])
                return round(kb / (1024 * 1024), 1)
    except Exception:
        pass
    return 0.0


def get_gpu_memory_gb() -> tuple[float, float]:
    """
    Get GPU memory (total, free) in GB.
    On Jetson (unified memory), this returns the tegra stats.
    """
    # Try nvidia-smi first (discrete GPU or Jetson with it)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,nounits,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            total = float(parts[0].strip()) / 1024  # MB to GB
            free = float(parts[1].strip()) / 1024
            return round(total, 1), round(free, 1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Jetson: check /sys/devices/gpu.0 or tegrastats
    # Fallback: Jetson shares system RAM
    total_ram = get_total_ram_gb()
    if total_ram > 0:
        # On Jetson, ~60-70% of RAM can be used as GPU memory
        return round(total_ram * 0.65, 1), round(get_available_ram_gb() * 0.5, 1)

    return 0.0, 0.0


def detect_jetson_board() -> str:
    """
    Detect which Jetson board we're running on.

    Returns a key from JETSON_PROFILES.
    """
    # Check /proc/device-tree/model (Jetson-specific)
    model_str = _read_proc_file("/proc/device-tree/model").lower()

    if "orin nano" in model_str:
        ram = get_total_ram_gb()
        if ram < 6:
            return "orin_nano_4gb"
        return "orin_nano_8gb"

    if "orin nx" in model_str:
        ram = get_total_ram_gb()
        if ram < 12:
            return "orin_nx_8gb"
        return "orin_nx_16gb"

    if "agx orin" in model_str:
        ram = get_total_ram_gb()
        if ram > 48:
            return "agx_orin_64gb"
        return "agx_orin_32gb"

    if "xavier nx" in model_str or "xavier-nx" in model_str:
        return "xavier_nx"

    if "nano" in model_str and "jetson" in model_str:
        return "nano_4gb"

    # Check for Jetson via tegra
    if os.path.exists("/etc/nv_tegra_release") or "tegra" in model_str:
        ram = get_total_ram_gb()
        if ram >= 16:
            return "orin_nx_16gb"
        elif ram >= 8:
            return "orin_nano_8gb"
        else:
            return "orin_nano_4gb"

    # Not a Jetson — check for GPU
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=5,
        )
        if result.returncode == 0:
            return "generic_gpu"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return "cpu_only"


def get_board_profile() -> JetsonProfile:
    """Detect the board and return its profile."""
    board_key = detect_jetson_board()
    profile = JETSON_PROFILES.get(board_key, JETSON_PROFILES["cpu_only"])
    logger.info("Detected board: %s (%s)", profile.name, board_key)
    return profile


def recommend_model_for_available_memory() -> dict:
    """
    Check available RAM and recommend the best model that will fit.

    Returns a dict with model info and recommended settings, or the
    profile defaults if detection fails.
    """
    available = get_available_ram_gb()
    total = get_total_ram_gb()

    if available <= 0:
        # Can't detect — use profile defaults
        profile = get_board_profile()
        return {
            "model": profile.recommended_llm,
            "url": profile.recommended_llm_url,
            "gpu_layers": profile.recommended_gpu_layers,
            "context_size": profile.recommended_context_size,
            "stt_model": profile.recommended_stt_model,
            "stt_compute": profile.recommended_stt_compute,
            "reason": "Could not detect available memory. Using profile defaults.",
        }

    logger.info("System RAM: %.1f GB total, %.1f GB available", total, available)

    # Reserve ~2GB for system + UI + other modules
    usable = available - 2.0
    if usable < 0.5:
        usable = 0.5

    # Find the best model that fits
    for tier in MODEL_TIERS:
        # Model needs roughly 1.2x its size in RAM (overhead)
        required = tier["size_gb"] * 1.2
        if usable >= required:
            logger.info(
                "Recommended model: %s (%.1f GB, need %.1f GB, have %.1f GB usable)",
                tier["name"], tier["size_gb"], required, usable,
            )
            # Adjust GPU layers based on available memory
            gpu_layers = 20
            if usable < 4:
                gpu_layers = 10
            elif usable > 8:
                gpu_layers = 33

            return {
                "model": tier["filename"],
                "url": tier["url"],
                "gpu_layers": gpu_layers,
                "context_size": min(tier["context"], 2048 if usable < 6 else 4096),
                "stt_model": "tiny" if usable < 4 else "small",
                "stt_compute": "int8" if usable < 4 else "float16",
                "reason": f"Selected {tier['name']} based on {usable:.1f} GB usable RAM.",
            }

    # Nothing fits — absolute minimum
    logger.warning("Very low memory (%.1f GB usable). Using minimum config.", usable)
    return {
        "model": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "url": MODEL_TIERS[-1]["url"],
        "gpu_layers": 0,
        "context_size": 512,
        "stt_model": "tiny",
        "stt_compute": "int8",
        "reason": f"Very low memory ({usable:.1f} GB). Using minimum viable config.",
    }


def print_system_info() -> None:
    """Print system info summary (for setup scripts and debugging)."""
    profile = get_board_profile()
    total = get_total_ram_gb()
    available = get_available_ram_gb()
    gpu_total, gpu_free = get_gpu_memory_gb()
    rec = recommend_model_for_available_memory()

    print("=" * 60)
    print("  EdgeTutor AI — System Information")
    print("=" * 60)
    print(f"  Board:          {profile.name}")
    print(f"  GPU:            {profile.gpu_name}")
    print(f"  CUDA cores:     {profile.cuda_cores}")
    print(f"  RAM:            {total:.1f} GB total, {available:.1f} GB available")
    print(f"  GPU memory:     {gpu_total:.1f} GB total, {gpu_free:.1f} GB free (shared)")
    print(f"  Unified memory: {'Yes' if profile.gpu_ram_shared else 'No'}")
    print()
    print("  Recommended configuration:")
    print(f"    LLM model:      {rec['model']}")
    print(f"    GPU layers:     {rec['gpu_layers']}")
    print(f"    Context size:   {rec['context_size']}")
    print(f"    STT model:      {rec['stt_model']}")
    print(f"    STT compute:    {rec['stt_compute']}")
    print(f"    Reason:         {rec['reason']}")
    print(f"  Notes: {profile.notes}")
    print("=" * 60)
