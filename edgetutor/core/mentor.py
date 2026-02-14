"""
EdgeTutor AI — AI Mentor Mode.

The AI Mentor teaches students about AI systems, GPUs, CUDA, LLMs,
quantization, and the very technology running on their Jetson device.
It turns the device itself into a learning lab.
"""

from __future__ import annotations

from edgetutor.core.logging_config import get_logger
from edgetutor.core.jetson import get_board_profile, get_total_ram_gb, get_available_ram_gb

logger = get_logger(__name__)

# ── Mentor system prompt ──────────────────────────────────────────────────────
MENTOR_SYSTEM_PROMPT = """\
You are EdgeTutor in AI Mentor mode. You teach students about artificial \
intelligence, machine learning, GPUs, and the technology running right on \
this device.

You are running on a real NVIDIA Jetson device. The student can ask you:
- How GPUs work and why they matter for AI
- What CUDA is and how parallel processing helps
- How Large Language Models (LLMs) work
- What quantization is and why it matters on edge devices
- How speech recognition (Whisper) works
- How OCR extracts text from images
- How vector search (RAG) finds relevant information
- What neural networks are and how they learn

{age_tone}

IMPORTANT RULES:
- You are running locally on a Jetson device with no internet. \
  Never suggest visiting websites.
- Use the real system stats below to make explanations concrete and tangible.
- Encourage curiosity. This is their AI lab!
- Be accurate but accessible. Use analogies.
- If they ask about your own architecture, explain it honestly and excitedly.

CURRENT SYSTEM STATS:
{system_stats}
"""

# ── Mentor topic library ──────────────────────────────────────────────────────
MENTOR_TOPICS = {
    "gpu": {
        "title": "How GPUs Work",
        "prompt": (
            "Explain how a GPU (Graphics Processing Unit) works differently from "
            "a CPU. Use analogies appropriate for the student's age. Mention that "
            "this Jetson device has a real NVIDIA GPU with CUDA cores."
        ),
    },
    "cuda": {
        "title": "What is CUDA?",
        "prompt": (
            "Explain CUDA — NVIDIA's parallel computing platform. Describe how "
            "it lets the GPU do many calculations at once, like having thousands "
            "of tiny workers instead of a few fast ones. Reference the actual "
            "CUDA cores on this Jetson device."
        ),
    },
    "llm": {
        "title": "How LLMs Work",
        "prompt": (
            "Explain how Large Language Models (LLMs) work at a high level. "
            "Cover: training on text data, tokens, prediction of next words, "
            "and how the model running on this device generates responses. "
            "Mention that the student is talking to a real LLM right now!"
        ),
    },
    "quantization": {
        "title": "What is Quantization?",
        "prompt": (
            "Explain model quantization — why we shrink AI models to run on "
            "small devices like Jetson. Use an analogy (e.g., compressing a "
            "photo). Mention the actual model size and quantization level "
            "being used on this device."
        ),
    },
    "whisper": {
        "title": "How Speech Recognition Works",
        "prompt": (
            "Explain how the Whisper speech-to-text model converts voice to "
            "text. Cover: audio → spectrogram → neural network → text. "
            "The student can test it by speaking into the microphone!"
        ),
    },
    "ocr": {
        "title": "How OCR Works",
        "prompt": (
            "Explain Optical Character Recognition (OCR) — how computers read "
            "text from images. Cover: image preprocessing, character detection, "
            "pattern matching. The student can test it by holding up a page!"
        ),
    },
    "rag": {
        "title": "How AI Search (RAG) Works",
        "prompt": (
            "Explain Retrieval-Augmented Generation (RAG) — how the tutor "
            "searches curriculum packs to find relevant information before "
            "answering. Cover: embeddings, vector similarity, and why this "
            "makes AI answers more accurate."
        ),
    },
    "neural_networks": {
        "title": "What are Neural Networks?",
        "prompt": (
            "Explain neural networks at an age-appropriate level. Use the "
            "analogy of brain neurons. Cover: layers, weights, learning from "
            "examples, and how they can recognize patterns (images, speech, text)."
        ),
    },
    "edge_ai": {
        "title": "What is Edge AI?",
        "prompt": (
            "Explain Edge AI — running AI models directly on devices instead "
            "of in the cloud. Cover: privacy (data stays local), speed (no "
            "internet latency), and independence (works offline). This Jetson "
            "device IS edge AI!"
        ),
    },
    "this_device": {
        "title": "About This Device",
        "prompt": (
            "Describe the Jetson device the student is using right now. "
            "Use the system stats to explain what each component does: "
            "GPU, RAM, CUDA cores. Explain how all the AI modules work "
            "together: voice → text → AI brain → voice response."
        ),
    },
}


def get_system_stats_text() -> str:
    """Generate a human-readable summary of current system stats."""
    try:
        profile = get_board_profile()
        total_ram = get_total_ram_gb()
        avail_ram = get_available_ram_gb()

        from edgetutor.core.settings import get_settings
        cfg = get_settings()

        stats = [
            f"Device: {profile.name}",
            f"GPU: {profile.gpu_name}",
            f"CUDA Cores: {profile.cuda_cores}",
            f"RAM: {total_ram:.1f} GB total, {avail_ram:.1f} GB available",
            f"Memory type: Unified (shared between CPU and GPU)",
            f"LLM model: {cfg.llm_model_path}",
            f"LLM GPU layers: {cfg.llm_n_gpu_layers}",
            f"LLM context window: {cfg.llm_context_size} tokens",
            f"STT model: Whisper {cfg.stt_model_size}",
            f"TTS: Piper (offline voice synthesis)",
            f"OCR: Tesseract (image text extraction)",
        ]
        return "\n".join(stats)
    except Exception as e:
        logger.warning("Could not get system stats: %s", e)
        return "System stats unavailable."


def build_mentor_prompt(age: str = "10") -> str:
    """Build the AI Mentor system prompt with current system stats."""
    from edgetutor.core.prompts import AGE_TONES

    age_tone = AGE_TONES.get(age, AGE_TONES["10"])
    stats = get_system_stats_text()

    return MENTOR_SYSTEM_PROMPT.format(
        age_tone=age_tone,
        system_stats=stats,
    ).strip()


def get_mentor_topic_list() -> list[dict]:
    """Return the list of mentor topics for the UI."""
    return [
        {"key": key, "title": info["title"]}
        for key, info in MENTOR_TOPICS.items()
    ]


def get_mentor_topic_prompt(topic_key: str) -> str:
    """Get the prompt for a specific mentor topic."""
    topic = MENTOR_TOPICS.get(topic_key)
    if topic:
        return topic["prompt"]
    return f"The student wants to learn about: {topic_key}"
