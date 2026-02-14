"""
EdgeTutor AI — Gradio Web UI.

Local web interface served on LAN (default localhost:7860).
Supports: chat, push-to-talk voice, camera/image capture, settings panel.
"""

from __future__ import annotations

import time
from typing import Optional

import gradio as gr

from edgetutor.core.logging_config import get_logger
from edgetutor.core.settings import get_settings

logger = get_logger(__name__)

# ── Global orchestrator reference (set during build) ──────────────────────────
_orchestrator = None
_module_status = {}


def _get_module_status_html() -> str:
    """Format module status as colored HTML badges."""
    badges = []
    for module, status in _module_status.items():
        if "ready" in status.lower():
            color = "#4caf50"
            icon = "✅"
        elif "disabled" in status.lower() or "not loaded" in status.lower():
            color = "#ff9800"
            icon = "⚠️"
        else:
            color = "#f44336"
            icon = "❌"
        badges.append(
            f'<span style="background:{color}22;color:{color};padding:2px 8px;'
            f'border-radius:12px;font-size:0.85em;margin:2px">'
            f"{icon} {module}: {status}</span>"
        )
    return " ".join(badges)


def _chat_respond(
    message: str,
    audio: Optional[tuple] = None,
    image: Optional[object] = None,
    history: list = None,
    age_mode: str = "10",
    subject_mode: str = "general",
    parent_mode: bool = False,
    quiz_mode: bool = False,
):
    """
    Main chat handler. Processes text, audio, and/or image input.
    Returns updated chat history.
    """
    if history is None:
        history = []

    if _orchestrator is None:
        history.append({"role": "assistant", "content": "⚠️ System not initialized. Please restart."})
        return history, history, None, ""

    from edgetutor.app.orchestrator import TutorRequest

    # Build request
    request = TutorRequest(
        user_text=message or "",
        settings_override={
            "age_mode": age_mode,
            "subject_mode": subject_mode,
            "parent_mode": parent_mode,
            "quiz_mode": quiz_mode,
        },
    )

    # Handle audio input
    if audio is not None:
        try:
            import numpy as np

            sr, audio_data = audio
            request.audio_array = audio_data
            request.audio_sample_rate = sr
        except Exception as e:
            logger.error("Audio processing error: %s", e)

    # Handle image input
    if image is not None:
        request.image = image

    # Build conversation history for context
    conv_history = []
    for msg in (history or [])[-10:]:  # Keep last 10 messages for context
        if isinstance(msg, dict):
            conv_history.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })

    # Process request
    response = _orchestrator.process(request, conversation_history=conv_history)

    # Determine what the user said (for display)
    display_text = message
    if not display_text and request.audio_array is not None:
        display_text = "🎤 [Voice input]"
    if image is not None:
        display_text = (display_text or "") + " 📷 [Image attached]"
    if not display_text:
        display_text = "..."

    # Update history
    history.append({"role": "user", "content": display_text.strip()})

    # Format response with extras
    response_text = response.text
    if response.ocr_text:
        response_text = f"📝 **Extracted text:**\n> {response.ocr_text[:300]}{'...' if len(response.ocr_text) > 300 else ''}\n\n{response_text}"
    if response.has_math and response.math_expressions:
        math_preview = "\n".join(f"- `{expr}`" for expr in response.math_expressions[:5])
        response_text = f"🔢 **Math detected:**\n{math_preview}\n\n{response_text}"

    # Add latency info (small footer)
    latency_parts = []
    for k, v in response.latency.items():
        latency_parts.append(f"{k}: {v:.1f}s")
    if latency_parts:
        response_text += f"\n\n<small>⏱️ {' | '.join(latency_parts)}</small>"

    history.append({"role": "assistant", "content": response_text})

    return history, history, response.audio, ""


def _scan_worksheet(
    image: Optional[object],
    history: list,
    age_mode: str,
    subject_mode: str,
    parent_mode: bool,
):
    """Handle 'Scan Worksheet' button click."""
    if image is None:
        history = history or []
        history.append({"role": "assistant", "content": "📷 Please capture or upload an image first, then click 'Scan Worksheet'."})
        return history, history, None

    return _chat_respond(
        message="Please explain this worksheet step by step.",
        image=image,
        history=history,
        age_mode=age_mode,
        subject_mode=subject_mode,
        parent_mode=parent_mode,
        quiz_mode=False,
    )[:3]


def _explain_step_by_step(
    history: list,
    age_mode: str,
    subject_mode: str,
    parent_mode: bool,
):
    """Handle 'Explain Step-by-Step' button."""
    if not history:
        history = [{"role": "assistant", "content": "There's nothing to explain yet! Ask me a question first."}]
        return history, history

    # Get the last user message to re-explain
    last_user = ""
    for msg in reversed(history):
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_user = msg.get("content", "")
            break

    if not last_user:
        history.append({"role": "assistant", "content": "Ask me a question first, and I'll explain it step by step!"})
        return history, history

    result = _chat_respond(
        message=f"Please explain this step by step in detail: {last_user}",
        history=history,
        age_mode=age_mode,
        subject_mode=subject_mode,
        parent_mode=parent_mode,
    )
    return result[0], result[1]


def build_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks interface."""
    global _orchestrator, _module_status

    from edgetutor.app.orchestrator import get_orchestrator

    _orchestrator = get_orchestrator()
    _module_status = _orchestrator.load_modules()

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="cyan",
        neutral_hue="slate",
    )

    with gr.Blocks(
        title="EdgeTutor AI",
        theme=theme,
        css="""
        .edgetutor-header { text-align: center; margin-bottom: 10px; }
        .edgetutor-header h1 { color: #1976d2; margin: 0; }
        .edgetutor-header p { color: #666; margin: 5px 0; }
        .status-bar { font-size: 0.85em; padding: 8px; background: #f5f5f5;
                      border-radius: 8px; margin-bottom: 10px; }
        footer { display: none !important; }
        """,
    ) as demo:
        # State
        chat_state = gr.State([])

        # Header
        gr.HTML(
            """
            <div class="edgetutor-header">
                <h1>🎓 EdgeTutor AI</h1>
                <p>Your offline AI study buddy — powered by NVIDIA Jetson</p>
            </div>
            """
        )

        # Module status bar
        status_html = gr.HTML(
            f'<div class="status-bar">{_get_module_status_html()}</div>'
        )

        with gr.Row():
            # ── LEFT: Chat + Input ────────────────────────────────────────
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="EdgeTutor",
                    height=480,
                    type="messages",
                    show_copy_button=True,
                    avatar_images=(None, "https://em-content.zobj.net/source/twitter/376/graduation-cap_1f393.png"),
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Ask EdgeTutor",
                        placeholder="Type your question here...",
                        scale=4,
                        lines=1,
                    )
                    send_btn = gr.Button("Send 📤", variant="primary", scale=1)

                with gr.Row():
                    audio_input = gr.Audio(
                        label="🎤 Push-to-Talk",
                        sources=["microphone"],
                        type="numpy",
                        scale=2,
                    )
                    audio_output = gr.Audio(
                        label="🔊 Tutor Response",
                        type="numpy",
                        autoplay=True,
                        scale=2,
                    )

            # ── RIGHT: Camera + Settings ──────────────────────────────────
            with gr.Column(scale=2):
                with gr.Accordion("📷 Camera / Image", open=True):
                    image_input = gr.Image(
                        label="Capture or upload worksheet",
                        sources=["webcam", "upload"],
                        type="pil",
                        height=250,
                    )
                    with gr.Row():
                        scan_btn = gr.Button("📄 Scan Worksheet", variant="secondary")
                        explain_btn = gr.Button("📝 Explain Step-by-Step", variant="secondary")

                with gr.Accordion("⚙️ Settings", open=False):
                    age_slider = gr.Radio(
                        choices=["7", "10", "16"],
                        value="10",
                        label="Age Mode",
                        info="Adjusts tone and explanation depth",
                    )
                    subject_dropdown = gr.Dropdown(
                        choices=["math", "reading", "science", "general"],
                        value="general",
                        label="Subject Mode",
                    )
                    parent_toggle = gr.Checkbox(
                        label="🔓 Parent Mode (direct answers)",
                        value=False,
                    )
                    quiz_toggle = gr.Checkbox(
                        label="📝 Quiz Mode (generate quizzes)",
                        value=False,
                    )

                with gr.Accordion("ℹ️ About", open=False):
                    gr.Markdown(
                        """
                        **EdgeTutor AI v0.1.0**

                        - 🔒 Fully offline — no data leaves this device
                        - 🛡️ Kid-safe guardrails enabled
                        - 🧠 Powered by local LLM on NVIDIA Jetson
                        - 📚 Add curriculum packs in the `content/` folder

                        [GitHub](https://github.com/thatcooperguy/edgetutor-ai)
                        """
                    )

        # ── Event handlers ────────────────────────────────────────────────
        send_inputs = [
            msg_input, audio_input, image_input, chat_state,
            age_slider, subject_dropdown, parent_toggle, quiz_toggle,
        ]
        send_outputs = [chatbot, chat_state, audio_output, msg_input]

        send_btn.click(
            fn=_chat_respond,
            inputs=send_inputs,
            outputs=send_outputs,
        )

        msg_input.submit(
            fn=_chat_respond,
            inputs=send_inputs,
            outputs=send_outputs,
        )

        scan_btn.click(
            fn=_scan_worksheet,
            inputs=[image_input, chat_state, age_slider, subject_dropdown, parent_toggle],
            outputs=[chatbot, chat_state, audio_output],
        )

        explain_btn.click(
            fn=_explain_step_by_step,
            inputs=[chat_state, age_slider, subject_dropdown, parent_toggle],
            outputs=[chatbot, chat_state],
        )

    return demo


def launch_ui() -> None:
    """Build and launch the Gradio UI."""
    cfg = get_settings()
    demo = build_ui()
    logger.info("Launching EdgeTutor UI on %s:%d", cfg.host, cfg.port)
    demo.launch(
        server_name=cfg.host,
        server_port=cfg.port,
        share=False,  # Offline device — no share link
        show_error=True,
    )
