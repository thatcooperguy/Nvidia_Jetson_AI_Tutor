"""
EdgeTutor AI — Gradio Web UI.

Local web interface served on LAN (default localhost:7860).
Dark theme. Supports: AI Tutor (streaming), AI Mentor, camera/voice, settings.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import gradio as gr

from edgetutor.core.conversations import get_conversation_store
from edgetutor.core.logging_config import get_logger
from edgetutor.core.settings import get_settings

if TYPE_CHECKING:
    from PIL import Image

logger = get_logger(__name__)

# ── Global state ──────────────────────────────────────────────────────────────
_orchestrator = None
_module_status = {}
_conversation_store = None


def _get_module_status_html() -> str:
    """Format module status as colored HTML badges for dark theme."""
    badges = []
    for module, status in _module_status.items():
        if "ready" in status.lower():
            color = "#76ff03"
            icon = "✅"
        elif "disabled" in status.lower() or "not loaded" in status.lower():
            color = "#ffc107"
            icon = "⚠️"
        else:
            color = "#ff5252"
            icon = "❌"
        badges.append(
            f'<span style="background:{color}15;color:{color};padding:3px 10px;'
            f'border-radius:14px;font-size:0.8em;margin:2px;border:1px solid {color}33">'
            f"{icon} <strong>{module}</strong>: {status}</span>"
        )
    return " ".join(badges)


def _get_system_info_html() -> str:
    """Generate system info panel HTML for AI Mentor mode."""
    try:
        from edgetutor.core.jetson import get_full_system_info

        info = get_full_system_info()
        return f"""
        <div style="font-family:monospace;font-size:0.85em;line-height:1.6;
                    background:#1a1a2e;padding:16px;border-radius:10px;border:1px solid #333">
            <div style="color:#76ff03;font-weight:bold;margin-bottom:8px">
                🖥️ YOUR AI LAB — System Stats
            </div>
            <table style="width:100%;color:#ccc">
                <tr><td style="color:#888">Board</td><td><strong>{info["board_name"]}</strong></td></tr>
                <tr><td style="color:#888">GPU</td><td>{info["gpu_name"]}</td></tr>
                <tr><td style="color:#888">CUDA Cores</td><td>{info["cuda_cores"]}</td></tr>
                <tr><td style="color:#888">CPU Cores</td><td>{info["cpu_cores"]}</td></tr>
                <tr><td style="color:#888">RAM</td><td>{info["ram_total_gb"]:.1f} GB total / {info["ram_available_gb"]:.1f} GB free</td></tr>
                <tr><td style="color:#888">GPU Memory</td><td>{info["gpu_mem_total_gb"]:.1f} GB (shared)</td></tr>
                <tr><td style="color:#888">Power Mode</td><td>{info["power_mode"]}</td></tr>
                <tr><td style="color:#888">Tegra</td><td>{"Yes ✓" if info["is_tegra"] else "No"}</td></tr>
                <tr><td colspan="2" style="border-top:1px solid #333;padding-top:6px"></td></tr>
                <tr><td style="color:#888">Active LLM</td><td style="color:#76ff03">{info["recommended_model"]}</td></tr>
                <tr><td style="color:#888">GPU Layers</td><td>{info["recommended_gpu_layers"]}</td></tr>
                <tr><td style="color:#888">Context</td><td>{info["recommended_context"]} tokens</td></tr>
                <tr><td style="color:#888">STT Model</td><td>Whisper {info["recommended_stt"]}</td></tr>
                <tr><td style="color:#888">Scaling</td><td style="color:#ffc107;font-size:0.85em">{info["scaling_reason"]}</td></tr>
            </table>
        </div>
        """
    except Exception as e:
        logger.warning("Could not generate system info HTML: %s", e)
        return '<div style="color:#888">System info unavailable.</div>'


def _build_conv_history(history: list | None, limit: int | None = None) -> list[dict]:
    """Extract the last *limit* messages as [{role, content}] dicts."""
    if limit is None:
        limit = get_settings().conversation_history_limit
    conv = []
    for msg in (history or [])[-limit:]:
        if isinstance(msg, dict):
            conv.append(
                {
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                }
            )
    return conv


def _auto_save(session_id: str, history: list) -> None:
    """Auto-save conversation history if store is available."""
    if _conversation_store and session_id and history:
        try:
            _conversation_store.save(session_id, history)
        except Exception as e:
            logger.debug("Auto-save failed for %s: %s", session_id, e)


def _format_errors(errors: list[str]) -> str:
    """Format orchestrator errors as a small HTML block."""
    if not errors:
        return ""
    lines = "\n".join(f"- {e}" for e in errors)
    return f"\n\n<small>⚠️ **Warnings:**\n{lines}</small>"


# ── Safety filter replacement sentinel ────────────────────────────────────────
_SAFETY_REPLACE_MARKER = "\n\n[Response replaced by safety filter]"


# ── Streaming chat handler ───────────────────────────────────────────────────
def _chat_respond_stream(
    message: str,
    audio: tuple | None = None,
    image: Image.Image | None = None,
    history: list | None = None,
    session_id: str = "",
    age_mode: str = "10",
    subject_mode: str = "general",
    parent_mode: bool = False,
    quiz_mode: bool = False,
):
    """Streaming chat handler for AI Tutor mode.

    This is a *generator* — Gradio calls it and yields partial chat
    histories so the user sees tokens appear in real time.
    """
    if history is None:
        history = []

    if _orchestrator is None:
        history.append({"role": "assistant", "content": "⚠️ System not initialized."})
        yield history, history, session_id, None, ""
        return

    from edgetutor.app.orchestrator import TutorRequest

    request = TutorRequest(
        user_text=message or "",
        settings_override={
            "age_mode": age_mode,
            "subject_mode": subject_mode,
            "parent_mode": parent_mode,
            "quiz_mode": quiz_mode,
        },
    )

    if audio is not None:
        try:
            sr, audio_data = audio
            request.audio_array = audio_data
            request.audio_sample_rate = sr
        except Exception as e:
            logger.error("Audio processing error: %s", e)

    if image is not None:
        request.image = image

    conv_history = _build_conv_history(history)

    # ── Add user message to history ──────────────────────────────────
    display_text = message
    if not display_text and request.audio_array is not None:
        display_text = "🎤 [Voice input]"
    if image is not None:
        display_text = (display_text or "") + " 📷 [Image attached]"
    if not display_text:
        display_text = "..."

    history.append({"role": "user", "content": display_text.strip()})

    # ── Stream tokens from the orchestrator ──────────────────────────
    accumulated = ""
    history.append({"role": "assistant", "content": ""})

    for token in _orchestrator.process_stream(request, conversation_history=conv_history):
        # Handle safety filter replacement — replace the whole message
        if _SAFETY_REPLACE_MARKER in token:
            from edgetutor.core.safety import get_redirect_message

            history[-1]["content"] = get_redirect_message()
            yield history, history, session_id, None, ""
            continue

        accumulated += token
        history[-1]["content"] = accumulated
        yield history, history, session_id, None, ""

    # Auto-save conversation
    _auto_save(session_id, history)

    # Final yield with complete text (no audio for streaming path)
    yield history, history, session_id, None, ""


# ── Non-streaming fallback (used by worksheet scanner, step-by-step) ─────────
def _chat_respond(
    message: str,
    image: Image.Image | None = None,
    history: list | None = None,
    session_id: str = "",
    age_mode: str = "10",
    subject_mode: str = "general",
    parent_mode: bool = False,
    quiz_mode: bool = False,
):
    """Non-streaming chat handler (used for worksheet scans, etc.)."""
    if history is None:
        history = []

    if _orchestrator is None:
        history.append({"role": "assistant", "content": "⚠️ System not initialized."})
        return history, history, session_id, None

    from edgetutor.app.orchestrator import TutorRequest

    request = TutorRequest(
        user_text=message or "",
        settings_override={
            "age_mode": age_mode,
            "subject_mode": subject_mode,
            "parent_mode": parent_mode,
            "quiz_mode": quiz_mode,
        },
    )

    if image is not None:
        request.image = image

    conv_history = _build_conv_history(history)
    response = _orchestrator.process(request, conversation_history=conv_history)

    display_text = message
    if image is not None:
        display_text = (display_text or "") + " 📷 [Image attached]"
    if not display_text:
        display_text = "..."

    history.append({"role": "user", "content": display_text.strip()})

    response_text = response.text
    if response.ocr_text:
        preview = response.ocr_text[:300]
        ellipsis = "..." if len(response.ocr_text) > 300 else ""
        response_text = f"📝 **Extracted text:**\n> {preview}{ellipsis}\n\n{response_text}"
    if response.has_math and response.math_expressions:
        math_preview = "\n".join(f"- `{expr}`" for expr in response.math_expressions[:5])
        response_text = f"🔢 **Math detected:**\n{math_preview}\n\n{response_text}"

    # Surface errors from orchestrator
    response_text += _format_errors(response.errors)

    latency_parts = [f"{k}: {v:.1f}s" for k, v in response.latency.items()]
    if latency_parts:
        response_text += f"\n\n<small>⏱️ {' | '.join(latency_parts)}</small>"

    history.append({"role": "assistant", "content": response_text})

    # Auto-save conversation
    _auto_save(session_id, history)

    return history, history, session_id, response.audio


# ── Mentor chat handler ───────────────────────────────────────────────────────
def _mentor_respond(
    message: str,
    mentor_topic: str,
    history: list | None = None,
    session_id: str = "",
    age_mode: str = "10",
):
    """Chat handler for AI Mentor mode."""
    if history is None:
        history = []

    if _orchestrator is None or not _orchestrator.llm or not _orchestrator.llm.is_loaded:
        history.append({"role": "assistant", "content": "⚠️ LLM not loaded. Check models/ folder."})
        return history, history, session_id, ""

    from edgetutor.core.mentor import build_mentor_prompt, get_mentor_topic_prompt

    # If a topic button was clicked, use that topic's prompt
    actual_message = message
    if mentor_topic and mentor_topic != "none" and not message.strip():
        actual_message = get_mentor_topic_prompt(mentor_topic)
    elif not actual_message.strip():
        actual_message = "Tell me about this Jetson device and how AI works on it!"

    # Safety check on mentor input
    cfg = get_settings()
    if cfg.safety_enabled:
        from edgetutor.core.safety import check_input_safety, get_redirect_message

        is_safe, _category = check_input_safety(actual_message)
        if not is_safe:
            display_text = message if message.strip() else f"📖 [Topic: {mentor_topic}]"
            history.append({"role": "user", "content": display_text})
            history.append({"role": "assistant", "content": get_redirect_message()})
            return history, history, session_id, ""

    system_prompt = build_mentor_prompt(age=age_mode)
    conv_history = _build_conv_history(history)

    display_text = message if message.strip() else f"📖 [Topic: {mentor_topic}]"
    history.append({"role": "user", "content": display_text})

    response_text = _orchestrator.llm.generate(
        user_message=actual_message,
        system_prompt=system_prompt,
        conversation_history=conv_history,
    )

    history.append({"role": "assistant", "content": response_text})

    # Auto-save conversation
    _auto_save(session_id, history)

    return history, history, session_id, ""


# ── Worksheet scanner ─────────────────────────────────────────────────────────
def _scan_worksheet(image, history, session_id, age_mode, subject_mode, parent_mode):
    if image is None:
        history = history or []
        history.append({"role": "assistant", "content": "📷 Upload or capture an image first!"})
        return history, history, session_id, None

    return _chat_respond(
        message="Please explain this worksheet step by step.",
        image=image,
        history=history,
        session_id=session_id,
        age_mode=age_mode,
        subject_mode=subject_mode,
        parent_mode=parent_mode,
    )


def _explain_step_by_step(history, session_id, age_mode, subject_mode, parent_mode):
    if not history:
        history = [{"role": "assistant", "content": "Ask me a question first!"}]
        return history, history, session_id

    last_user = ""
    for msg in reversed(history):
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_user = msg.get("content", "")
            break

    if not last_user:
        history.append({"role": "assistant", "content": "Ask me a question first!"})
        return history, history, session_id

    result = _chat_respond(
        message=f"Please explain this step by step in detail: {last_user}",
        history=history,
        session_id=session_id,
        age_mode=age_mode,
        subject_mode=subject_mode,
        parent_mode=parent_mode,
    )
    return result[0], result[1], result[2]


# ── Build the full UI ─────────────────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    """Build the Gradio dark-themed interface with Tutor + Mentor tabs."""
    global _orchestrator, _module_status, _conversation_store

    from edgetutor.app.orchestrator import get_orchestrator

    _orchestrator = get_orchestrator()
    _module_status = _orchestrator.load_modules()
    _conversation_store = get_conversation_store()

    # Dark theme
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.slate,
    ).set(
        body_background_fill="#0f0f1a",
        body_background_fill_dark="#0f0f1a",
        block_background_fill="#1a1a2e",
        block_background_fill_dark="#1a1a2e",
        block_border_color="#2a2a4a",
        block_border_color_dark="#2a2a4a",
        input_background_fill="#16213e",
        input_background_fill_dark="#16213e",
        button_primary_background_fill="#1976d2",
        button_primary_background_fill_dark="#1976d2",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="#2a2a4a",
        button_secondary_background_fill_dark="#2a2a4a",
        button_secondary_text_color="#ffffff",
    )

    with gr.Blocks(
        title="EdgeTutor AI — Your AI Lab",
        theme=theme,
        css="""
        .header-box { text-align:center; padding:20px 10px 10px; }
        .header-box h1 { color:#64b5f6; margin:0; font-size:2em; }
        .header-box .subtitle { color:#90caf9; margin:4px 0; font-size:1.1em; }
        .header-box .tagline { color:#666; font-size:0.85em; margin-top:4px; }
        .status-bar { font-size:0.8em; padding:8px 12px; background:#1a1a2e;
                      border-radius:10px; border:1px solid #2a2a4a; }
        .mentor-topic-btn { min-height:60px !important; }
        footer { display:none !important; }
        .dark { --body-text-color: #e0e0e0; }
        """,
    ) as demo:
        # ── State ─────────────────────────────────────────────────────────
        tutor_state = gr.State([])
        mentor_state = gr.State([])
        tutor_session_id = gr.State(str(uuid.uuid4())[:8])
        mentor_session_id = gr.State(str(uuid.uuid4())[:8])

        # ── Header ────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="header-box">
            <h1>🎓 EdgeTutor AI</h1>
            <div class="subtitle">Welcome to Your AI Lab</div>
            <div class="tagline">
                Fully offline • Kid-safe • Powered by NVIDIA Jetson
            </div>
        </div>
        """)

        # ── Module status ─────────────────────────────────────────────────
        gr.HTML(f'<div class="status-bar">{_get_module_status_html()}</div>')

        # ── Tabs ──────────────────────────────────────────────────────────
        with gr.Tabs():
            # ══════════════ TAB 1: AI TUTOR ═══════════════════════════════
            with gr.TabItem("🎓 AI Tutor", id="tutor"):
                with gr.Row():
                    with gr.Column(scale=3):
                        tutor_chatbot = gr.Chatbot(
                            label="EdgeTutor",
                            height=440,
                            type="messages",
                            show_copy_button=True,
                        )
                        with gr.Row():
                            tutor_input = gr.Textbox(
                                label="Ask EdgeTutor",
                                placeholder="Type your question here...",
                                scale=4,
                                lines=1,
                            )
                            tutor_send = gr.Button("Send 📤", variant="primary", scale=1)

                        with gr.Row():
                            audio_input = gr.Audio(
                                label="🎤 Push-to-Talk",
                                sources=["microphone"],
                                type="numpy",
                                scale=2,
                            )
                            audio_output = gr.Audio(
                                label="🔊 Response",
                                type="numpy",
                                autoplay=True,
                                scale=2,
                            )

                    with gr.Column(scale=2):
                        with gr.Accordion("📷 Camera / Image", open=True):
                            image_input = gr.Image(
                                label="Capture or upload worksheet",
                                sources=["webcam", "upload"],
                                type="pil",
                                height=220,
                            )
                            with gr.Row():
                                scan_btn = gr.Button("📄 Scan Worksheet", variant="secondary")
                                explain_btn = gr.Button("📝 Step-by-Step", variant="secondary")

                        with gr.Accordion("⚙️ Settings", open=False):
                            age_slider = gr.Radio(
                                choices=["7", "10", "16"],
                                value="10",
                                label="Age Mode",
                                info="Adjusts tone and depth",
                            )
                            subject_dropdown = gr.Dropdown(
                                choices=["math", "reading", "science", "general"],
                                value="general",
                                label="Subject",
                            )
                            parent_toggle = gr.Checkbox(
                                label="🔓 Parent Mode (direct answers)",
                                value=False,
                            )
                            quiz_toggle = gr.Checkbox(
                                label="📝 Quiz Mode",
                                value=False,
                            )

                # Tutor event handlers — streaming for text, non-streaming for scans
                tutor_inputs = [
                    tutor_input,
                    audio_input,
                    image_input,
                    tutor_state,
                    tutor_session_id,
                    age_slider,
                    subject_dropdown,
                    parent_toggle,
                    quiz_toggle,
                ]
                tutor_outputs = [
                    tutor_chatbot,
                    tutor_state,
                    tutor_session_id,
                    audio_output,
                    tutor_input,
                ]

                tutor_send.click(
                    fn=_chat_respond_stream,
                    inputs=tutor_inputs,
                    outputs=tutor_outputs,
                )
                tutor_input.submit(
                    fn=_chat_respond_stream,
                    inputs=tutor_inputs,
                    outputs=tutor_outputs,
                )
                scan_btn.click(
                    fn=_scan_worksheet,
                    inputs=[
                        image_input,
                        tutor_state,
                        tutor_session_id,
                        age_slider,
                        subject_dropdown,
                        parent_toggle,
                    ],
                    outputs=[tutor_chatbot, tutor_state, tutor_session_id, audio_output],
                )
                explain_btn.click(
                    fn=_explain_step_by_step,
                    inputs=[
                        tutor_state,
                        tutor_session_id,
                        age_slider,
                        subject_dropdown,
                        parent_toggle,
                    ],
                    outputs=[tutor_chatbot, tutor_state, tutor_session_id],
                )

            # ══════════════ TAB 2: AI MENTOR ══════════════════════════════
            with gr.TabItem("🧠 AI Mentor", id="mentor"):
                gr.Markdown(
                    "### Learn How AI Works — On This Device!\n"
                    "Pick a topic below or ask your own question. "
                    "The AI Mentor will explain using the real hardware stats "
                    "from your Jetson device."
                )

                with gr.Row():
                    with gr.Column(scale=3):
                        mentor_chatbot = gr.Chatbot(
                            label="AI Mentor",
                            height=440,
                            type="messages",
                            show_copy_button=True,
                        )
                        with gr.Row():
                            mentor_input = gr.Textbox(
                                label="Ask the AI Mentor",
                                placeholder="How do GPUs help with AI?",
                                scale=4,
                                lines=1,
                            )
                            mentor_send = gr.Button("Ask 🧠", variant="primary", scale=1)

                    with gr.Column(scale=2):
                        gr.Markdown("#### 📖 Explore Topics")

                        topic_buttons = {}
                        topic_labels = {
                            "this_device": "🖥️ About This Device",
                            "gpu": "🎮 How GPUs Work",
                            "cuda": "⚡ What is CUDA?",
                            "llm": "🧠 How LLMs Work",
                            "quantization": "📦 What is Quantization?",
                            "neural_networks": "🕸️ Neural Networks",
                            "edge_ai": "📡 What is Edge AI?",
                            "whisper": "🎤 Speech Recognition",
                            "ocr": "👁️ How OCR Works",
                            "rag": "🔍 AI Search (RAG)",
                        }

                        for key, label in topic_labels.items():
                            btn = gr.Button(
                                label, variant="secondary", elem_classes=["mentor-topic-btn"]
                            )
                            topic_buttons[key] = btn

                        # Age selector for mentor
                        mentor_age = gr.Radio(
                            choices=["7", "10", "16"],
                            value="10",
                            label="Explanation Level",
                        )

                        # System info panel
                        gr.HTML(_get_system_info_html())

                # Mentor event handlers
                mentor_outputs = [
                    mentor_chatbot,
                    mentor_state,
                    mentor_session_id,
                    mentor_input,
                ]
                mentor_send.click(
                    fn=_mentor_respond,
                    inputs=[
                        mentor_input,
                        gr.State("none"),
                        mentor_state,
                        mentor_session_id,
                        mentor_age,
                    ],
                    outputs=mentor_outputs,
                )
                mentor_input.submit(
                    fn=_mentor_respond,
                    inputs=[
                        mentor_input,
                        gr.State("none"),
                        mentor_state,
                        mentor_session_id,
                        mentor_age,
                    ],
                    outputs=mentor_outputs,
                )

                # Topic button handlers
                for topic_key, btn in topic_buttons.items():
                    btn.click(
                        fn=_mentor_respond,
                        inputs=[
                            gr.State(""),
                            gr.State(topic_key),
                            mentor_state,
                            mentor_session_id,
                            mentor_age,
                        ],
                        outputs=mentor_outputs,
                    )

            # ══════════════ TAB 3: ABOUT ══════════════════════════════════
            with gr.TabItem("ℹ️ About", id="about"):
                from edgetutor import __version__

                gr.Markdown(f"""
### EdgeTutor AI v{__version__}

**An offline-first AI Tutor + AI Mentor for NVIDIA Jetson.**

| Feature | Status |
|---------|--------|
| 💬 Chat Tutor | ✅ Text-based tutoring |
| 🎤 Voice Input | ✅ Push-to-talk (Whisper) |
| 🔊 Voice Output | ✅ Text-to-speech (Piper) |
| 📷 Worksheet Scanner | ✅ OCR + explanation |
| 🔢 Math Detection | ✅ Auto-detect equations |
| 🧠 AI Mentor | ✅ Learn about AI/GPUs/CUDA |
| 📚 Content Packs | ✅ RAG with local docs |
| 🛡️ Kid-Safe | ✅ Content filtering |
| ⚡ Auto-Scaling | ✅ Adapts to your hardware |

---

**Privacy**: All processing happens locally on this device.
No data is sent anywhere. No telemetry. No tracking.

**Disclaimer**: This is an independent open-source project and is not
affiliated with or endorsed by NVIDIA.

[GitHub](https://github.com/thatcooperguy/Nvidia_Jetson_AI_Tutor) •
[MIT License](https://github.com/thatcooperguy/Nvidia_Jetson_AI_Tutor/blob/main/LICENSE)
                """)

    return demo


def launch_ui() -> None:
    """Build and launch the Gradio UI."""
    cfg = get_settings()
    demo = build_ui()
    logger.info("Launching EdgeTutor UI on %s:%d", cfg.host, cfg.port)
    demo.launch(
        server_name=cfg.host,
        server_port=cfg.port,
        share=False,
        show_error=True,
    )
