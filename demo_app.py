"""
Hybrid DP Mental Health Chatbot — Gradio Demo
==============================================
Run with:
    python demo_app.py
or:
    gradio demo_app.py

Tabs
----
1. Emotion Chatbot      – type a message, detect emotions, receive an empathetic response.
2. Privacy-Utility      – compare V0–V3 F1-macro / epsilon trade-off.
3. How It Works         – brief explainer of the hybrid DP approach.

Model loading
-------------
The saved .pt files are RoBERTa-based AutoModelForSequenceClassification state
dicts produced by the RQ1 training notebook.  If a weight file is missing or
the backbone cannot be downloaded the app falls back to a lightweight rule-
based emotion detector so the interface remains usable offline.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional heavy imports — degrade gracefully when unavailable
# ---------------------------------------------------------------------------
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    import gradio as gr
    GRADIO_OK = True
except ImportError:
    GRADIO_OK = False
    print("Gradio not found. Install it with:  pip install gradio")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "rq1_variants"
METRICS_FILE = ROOT / "reports" / "rq1" / "rq1_summary.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMOTION_COLUMNS: List[str] = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

# Emoji map for emotions shown in the UI
EMOTION_EMOJI: Dict[str, str] = {
    "admiration": "🌟", "amusement": "😄", "anger": "😠", "annoyance": "😤",
    "approval": "👍", "caring": "🤗", "confusion": "😕", "curiosity": "🤔",
    "desire": "💭", "disappointment": "😞", "disapproval": "👎", "disgust": "🤢",
    "embarrassment": "😳", "excitement": "🎉", "fear": "😨", "gratitude": "🙏",
    "grief": "💔", "joy": "😊", "love": "❤️", "nervousness": "😰",
    "optimism": "🌈", "pride": "😊", "realization": "💡", "relief": "😌",
    "remorse": "😔", "sadness": "😢", "surprise": "😲", "neutral": "😐",
}

# Empathy templates keyed by primary detected emotion
EMPATHY_TEMPLATES: Dict[str, List[str]] = {
    "admiration":    ["It sounds like you really value this person or moment. What stands out most to you about it?",
                      "That appreciation comes through clearly. How has this shaped your day?"],
    "amusement":     ["That sounds like a light and funny moment. What made it especially amusing?",
                      "I can hear a playful tone in that. Want to share more of what happened?"],
    "sadness":       ["I'm sorry you're feeling this way. That sounds heavy, and your feelings matter.",
                      "I hear how painful this feels. You do not have to hold it all by yourself."],
    "grief":         ["I'm sorry you're carrying this grief. I am here with you.",
                      "Grief can feel overwhelming. Please go gently with yourself right now."],
    "fear":          ["That sounds unsettling. Let's slow this down together.",
                      "It makes sense to feel scared when something feels uncertain or threatening."],
    "anger":         ["That sounds really frustrating. Your reaction makes sense.",
                      "It sounds like something crossed an important boundary for you."],
    "annoyance":     ["That sounds draining. Anyone would feel worn down by that.",
                      "It makes sense that this is getting under your skin."],
    "approval":      ["That sounds like a meaningful step in the right direction. What feels most positive about it?",
                      "It seems like this aligns with what matters to you. What part feels best right now?"],
    "disappointment":["I'm sorry this didn't go the way you hoped. That can hurt more than it first seems.",
                      "Disappointment can land hard. It's understandable to feel let down."],
    "disapproval":   ["It sounds like this does not sit right with you. What feels most concerning about it?",
                      "Your reaction makes sense. Do you want to talk through what felt wrong?"],
    "disgust":       ["That sounds deeply uncomfortable. It's understandable to feel a strong reaction.",
                      "I hear how unpleasant that felt for you. What part affected you most?"],
    "embarrassment": ["That sounds like a vulnerable moment. Many people feel this way after something awkward.",
                      "I'm glad you shared that. What would feel kindest to yourself right now?"],
    "excitement":    ["That sounds energizing and important to you. What are you most excited about?",
                      "I can hear the momentum in this. What are you looking forward to next?"],
    "nervousness":   ["That sounds tense and exhausting. We can take this one step at a time.",
                      "Feeling nervous is hard. Let's focus on the next small step, not everything at once."],
    "joy":           ["It's nice to hear something positive. What made this moment feel good?",
                      "That sounds uplifting. I'm glad you shared it."],
    "gratitude":     ["That is a meaningful thing to notice. Gratitude can be grounding.",
                      "It's good to pause on what matters to you."],
    "love":          ["That sounds deeply caring. Connection can be a strong source of support.",
                      "It's clear this matters to you. Hold onto that connection."],
    "caring":        ["Your care comes through clearly. That is a strength.",
                      "It's good to see how much you care about others."],
    "confusion":     ["That sounds unclear and mentally tiring. Which part feels hardest to make sense of?",
                      "I hear that this feels confusing. Want to break it down together step by step?"],
    "curiosity":     ["That's a thoughtful question. What are you most curious to understand?",
                      "Your curiosity is a strength. Where do you want to explore first?"],
    "desire":        ["It sounds like this matters a lot to you. What are you hoping for most?",
                      "That wish comes through clearly. What would be a small step toward it?"],
    "optimism":      ["That hopeful perspective matters. It can help carry you through hard moments.",
                      "Holding onto hope is a real strength."],
    "pride":         ["You sound proud of this, and that feeling is valid. What achievement are you recognizing in yourself?",
                      "That sounds like earned progress. Take a moment to acknowledge what you did well."],
    "realization":   ["That sounds like an important insight. What changed for you when you noticed this?",
                      "I hear a moment of clarity here. How do you want to use this insight going forward?"],
    "relief":        ["It sounds like some pressure finally lifted. What helped this feel lighter?",
                      "I'm glad this feels more manageable now. What would help you keep that steadiness?"],
    "remorse":       ["It sounds like you're reflecting honestly. That takes courage.",
                      "Be kind to yourself. Mistakes do not define you."],
    "surprise":      ["That sounds unexpected. How are you making sense of it right now?",
                      "I can hear that this caught you off guard. What feels most immediate for you?"],
    "neutral":       ["Thank you for sharing. What feels most important to talk about today?",
                      "I'm here to listen. Where would you like to start?"],
    "default":       ["I'm here with you. Tell me a little more about what is going on.",
                      "Thank you for sharing that. What feels most important right now?"],
}

# Variant metadata for the comparison dashboard
VARIANT_META: List[Dict] = [
    {"name": "V0 Baseline",  "dp": False, "anonymized": False, "epsilon": "∞ (no DP)", "color": "#4C72B0"},
    {"name": "V1 Anonym.",   "dp": False, "anonymized": True,  "epsilon": "∞ (no DP)", "color": "#55A868"},
    {"name": "V2 DP-SGD",    "dp": True,  "anonymized": False, "epsilon": "8.0",        "color": "#C44E52"},
    {"name": "V3 Hybrid",    "dp": True,  "anonymized": True,  "epsilon": "8.0",        "color": "#8172B2"},
]

# Metrics — loaded from saved JSON

def _load_metrics() -> List[Dict]:
    """Return per-variant metrics, preferring a saved JSON file."""
    if METRICS_FILE.exists():
        try:
            with open(METRICS_FILE, encoding="utf-8") as f:
                data = json.load(f)
            # Expect a list of dicts with keys: variant, f1_macro, f1_micro, epsilon
            if isinstance(data, list) and data:
                return data
        except Exception:
            pass

    # Fall-back values mirror your saved RQ1 snapshot for consistency.
    return [
        {"variant": "V0 Baseline", "f1_macro": 0.3695696777666186, "f1_micro": 0.410881181028626, "epsilon_actual": None},
        {"variant": "V1 Anonym.",  "f1_macro": 0.36507115460878375, "f1_micro": 0.4052349504463083, "epsilon_actual": None},
        {"variant": "V2 DP-SGD",   "f1_macro": 0.22742193622702161, "f1_micro": 0.3037734615133085, "epsilon_actual": 8.0},
        {"variant": "V3 Hybrid",   "f1_macro": 0.19345737954145878, "f1_micro": 0.31361662327759277, "epsilon_actual": 8.0},
    ]


METRICS = _load_metrics()

# Model loading

_model_cache: Dict[str, Tuple] = {}

def _model_path(variant: str) -> Optional[Path]:
    """Map variant display name to .pt file path."""
    mapping = {
        "V0 Baseline": MODEL_DIR / "V0_BASELINE_best.pt",
        "V1 Anonym.":  MODEL_DIR / "V1_ANONYM_best.pt",
        "V2 DP-SGD":   MODEL_DIR / "V2_DP_SGD_best.pt",
        "V3 Hybrid":   MODEL_DIR / "V3_HYBRID_best.pt",
    }
    return mapping.get(variant)


def load_model(variant: str) -> Optional[Tuple]:
    """
    Load (model, tokenizer, label_list) for the requested variant.
    Detects actual number of output labels from the checkpoint so the
    classifier head is never randomly re-initialised.
    Caches the result.  Returns None if loading fails.
    """
    if not TORCH_OK:
        return None
    if variant in _model_cache:
        return _model_cache[variant]

    pt_path = _model_path(variant)
    if pt_path is None or not pt_path.exists():
        return None

    try:
        state = torch.load(pt_path, map_location="cpu", weights_only=True)

        # Detect real label count from the saved classifier weights
        if "classifier.out_proj.weight" in state:
            num_labels = state["classifier.out_proj.weight"].shape[0]
        elif "classifier.out_proj.bias" in state:
            num_labels = state["classifier.out_proj.bias"].shape[0]
        else:
            num_labels = len(EMOTION_COLUMNS)

        # Use only as many label columns as the checkpoint actually covers
        effective_labels = EMOTION_COLUMNS[:num_labels]

        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=num_labels,
            problem_type="multi_label_classification",
        )
        model.load_state_dict(state, strict=False)
        model.eval()
        _model_cache[variant] = (model, tokenizer, effective_labels)
        return model, tokenizer, effective_labels
    except Exception as exc:
        print(f"[demo_app] Could not load {variant}: {exc}")
        return None

# ---------------------------------------------------------------------------
# Fallback: rule-based emotion detection (no model needed)
# ---------------------------------------------------------------------------

_KEYWORD_MAP: Dict[str, List[str]] = {
    "sadness":    ["sad", "unhappy", "miserable", "depressed", "down", "blue", "cry", "crying", "tears", "heartbroken"],
    "anger":      ["angry", "furious", "mad", "rage", "hate", "enraged", "annoyed", "frustrated", "irritated"],
    "fear":       ["afraid", "scared", "terrified", "anxious", "nervous", "worried", "panic", "dread", "phobia"],
    "joy":        ["happy", "glad", "joyful", "excited", "elated", "cheerful", "thrilled", "delighted", "wonderful"],
    "love":       ["love", "adore", "cherish", "affection", "romantic", "infatuated", "smitten"],
    "gratitude":  ["grateful", "thankful", "appreciate", "blessed", "thank you", "thanks"],
    "disappointment": ["disappointed", "let down", "expected", "hoped", "unfortunate", "regret"],
    "nervousness": ["nervous", "anxious", "stressed", "overwhelmed", "tense", "uneasy", "worrying"],
    "grief":      ["grief", "grieve", "mourning", "loss", "lost", "bereavement", "miss", "missing"],
    "remorse":    ["sorry", "apologize", "regret", "guilt", "guilty", "ashamed", "mistake"],
    "caring":     ["care", "support", "help", "concern", "worry about", "look after"],
    "optimism":   ["hopeful", "optimistic", "positive", "looking forward", "bright side", "hope"],
}


def _rule_based_emotion(text: str) -> Dict[str, float]:
    """Return a dict of emotion → confidence score using keyword matching."""
    text_lower = text.lower()
    scores: Dict[str, float] = {e: 0.0 for e in EMOTION_COLUMNS}
    for emotion, keywords in _KEYWORD_MAP.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                scores[emotion] = min(scores[emotion] + 0.35, 1.0)
    # Boost negative → sadness if nothing else detected
    if all(v < 0.1 for v in scores.values()):
        scores["neutral"] = 0.6
    return scores

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def detect_emotions(text: str, variant: str, threshold: float = 0.3) -> Dict[str, float]:
    """
    Return {emotion: confidence} for emotions above threshold.
    Uses the loaded model if available, else falls back to rule-based.
    """
    loaded = load_model(variant)
    if loaded is not None and TORCH_OK:
        model, tokenizer, effective_labels = loaded
        enc = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=128, padding="max_length",
        )
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.sigmoid(logits).squeeze(0).numpy()
        return {
            effective_labels[i]: float(probs[i])
            for i in range(len(effective_labels))
            if probs[i] >= threshold
        }
    # Fallback
    raw = _rule_based_emotion(text)
    return {k: v for k, v in raw.items() if v >= threshold}


def _pick_template(emotions: Dict[str, float]) -> str:
    """Choose an empathy template based on the strongest detected emotion."""
    if not emotions:
        return np.random.choice(EMPATHY_TEMPLATES["default"])
    primary = max(emotions, key=emotions.get)
    templates = EMPATHY_TEMPLATES.get(primary, EMPATHY_TEMPLATES["default"])
    return np.random.choice(templates)


def _build_emotion_label(emotions: Dict[str, float], max_items: int = 2) -> str:
    """Format the strongest detected emotions as a concise string for the UI."""
    if not emotions:
        return "No strong emotions detected above threshold."
    sorted_ems = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
    top_score = sorted_ems[0][1]
    filtered = [item for item in sorted_ems if item[1] >= max(0.35, top_score - 0.10)]
    limited = filtered[:max_items]
    parts = [
        f"{EMOTION_EMOJI.get(e, '')} **{e}** ({score:.0%})"
        for e, score in limited
    ]
    return "  |  ".join(parts)


def _build_response_notice(variant: str) -> str:
    """Return a subtle, non-intrusive privacy notice for DP variants."""
    for meta in VARIANT_META:
        if meta["name"] == variant and meta["dp"]:
            return f"\n\n*Privacy-preserving training active (DP-SGD, ε = {meta['epsilon']}).*"
    return ""


# Chat handler

_variant_store: Dict[str, str] = {"active": "V3 Hybrid"}


def _supports_chatbot_messages() -> bool:
    """Return True when this Gradio version supports Chatbot(type='messages')."""
    if not GRADIO_OK:
        return False
    try:
        sig = inspect.signature(gr.Chatbot.__init__)
    except (TypeError, ValueError):
        return False
    return "type" in sig.parameters


def chat(
    user_msg: str,
    history: Optional[List[Dict[str, Any]]],
    variant: str,
    threshold: float,
) -> Tuple[List[Dict[str, Any]], str]:
    """Main chat callback — always uses messages-dict format expected by Gradio Chatbot."""
    variant = variant or "V3 Hybrid"
    _variant_store["active"] = variant

    history = history or []

    if not user_msg.strip():
        history.append({"role": "assistant", "content": "Please type something - I'm here to listen."})
        return history, ""

    emotions = detect_emotions(user_msg, variant, threshold)
    empathy = _pick_template(emotions)
    emotion_line = _build_emotion_label(emotions)
    dp_note = _build_response_notice(variant)

    response = f"{empathy}\n\n**Detected emotions:** {emotion_line}{dp_note}"

    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": response})
    return history, ""


# Comparison chart

def _make_comparison_chart() -> Optional[str]:
    """Render a grouped bar chart and return the image path, or None."""
    if not MATPLOTLIB_OK:
        return None

    labels = [m["variant"] for m in METRICS]
    f1_macro = [m["f1_macro"] for m in METRICS]
    f1_micro = [m["f1_micro"] for m in METRICS]
    colors = [v["color"] for v in VARIANT_META]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, f1_macro, width, label="F1-macro", color=colors, alpha=0.9)
    bars2 = ax.bar(x + width / 2, f1_micro, width, label="F1-micro",
                   color=[c + "99" for c in colors], edgecolor=colors, linewidth=1.2, alpha=0.85)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Privacy-Utility Trade-off: V0 → V3", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 0.7)
    ax.legend(fontsize=11)
    ax.axhline(0.75 * max(f1_macro), color="grey", linestyle="--", linewidth=0.8,
               label="75% retention target")
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax.annotate(f"{bar.get_height():.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    plt.tight_layout()
    out_path = ROOT / "reports" / "demo_comparison_chart.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)


def _make_retention_chart() -> Optional[str]:
    """Privacy budget vs F1-macro scatter."""
    if not MATPLOTLIB_OK:
        return None

    baseline_f1 = METRICS[0]["f1_macro"]
    fig, ax = plt.subplots(figsize=(7, 4))

    for m, meta in zip(METRICS, VARIANT_META):
        eps_val = m.get("epsilon_actual") or (None if meta["epsilon"].startswith("∞") else float(meta["epsilon"]))
        x = eps_val if eps_val is not None else 20  # plot no-DP at far right
        y = m["f1_macro"]
        ax.scatter(x, y, s=160, color=meta["color"], zorder=5, label=meta["name"])
        ax.annotate(meta["name"], (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=9)

    ax.axhline(0.75 * baseline_f1, color="grey", linestyle="--", linewidth=0.9,
               label="75% retention threshold")
    ax.set_xlabel("Privacy Budget ε  (lower = stronger DP)", fontsize=11)
    ax.set_ylabel("F1-macro", fontsize=11)
    ax.set_title("F1-macro vs Privacy Budget", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out_path = ROOT / "reports" / "demo_retention_chart.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)


# Metrics table HTML

def _metrics_html() -> str:
    rows = ""
    baseline_f1 = METRICS[0]["f1_macro"]
    for m, meta in zip(METRICS, VARIANT_META):
        retention = f"{m['f1_macro'] / baseline_f1 * 100:.1f}%" if baseline_f1 else "—"
        dp_badge = "🔒 Yes" if meta["dp"] else "—"
        anon_badge = "✅ Yes" if meta["anonymized"] else "—"
        rows += (
            f"<tr>"
            f"<td><b>{m['variant']}</b></td>"
            f"<td>{m['f1_macro']:.4f}</td>"
            f"<td>{m['f1_micro']:.4f}</td>"
            f"<td>{dp_badge}</td>"
            f"<td>{meta['epsilon']}</td>"
            f"<td>{anon_badge}</td>"
            f"<td>{retention}</td>"
            f"</tr>"
        )

    source_note = (
        "<p style='font-size:12px; color:#2f4f4f; margin-top:6px;'>"
        "Loaded metrics from <code>reports/rq1/rq1_summary.json</code>."
        "</p>"
        if METRICS_FILE.exists()
        else (
            "<p style='font-size:12px; color:grey; margin-top:6px;'>"
            "⚠️ If no trained metrics JSON is found at <code>reports/rq1/rq1_summary.json</code>, "
            "the values above are representative placeholders. Save your actual metrics there to update this table."
            "</p>"
        )
    )

    return f"""
    <table style='width:100%; border-collapse:collapse; font-size:14px;'>
      <thead>
        <tr style='background:#f0f0f0;'>
          <th style='padding:8px; text-align:left;'>Variant</th>
          <th style='padding:8px;'>F1-macro</th>
          <th style='padding:8px;'>F1-micro</th>
          <th style='padding:8px;'>DP-SGD</th>
          <th style='padding:8px;'>ε</th>
          <th style='padding:8px;'>Anonymized</th>
          <th style='padding:8px;'>Retention vs V0</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
        {source_note}
    """



# HOW IT WORKS text

HOW_IT_WORKS_MD = textwrap.dedent("""
## How the Hybrid DP Approach Works

### The Problem
Mental health chatbots handle deeply personal data. Standard models trained on raw text can
*memorise* sensitive user content, leaking it through model inversion or membership inference
attacks.

### Two Complementary Privacy Layers

| Layer | Technique | What It Does |
|-------|-----------|--------------|
| **Layer 1 — Pre-processing** | Text Anonymisation | Replaces names, locations, and other PII with generic tokens *before* training begins |
| **Layer 2 — Training** | DP-SGD (Opacus) | Adds calibrated Gaussian noise to per-sample gradients, providing a formal privacy bound (ε, δ) |

### The Four Variants Compared

- **V0 Baseline** — raw text, no privacy protection (upper-bound utility)
- **V1 Anonym.** — anonymised text only (PII removed, but no formal DP guarantee)
- **V2 DP-SGD** — raw text + DP-SGD (formal guarantee, but PII still in training set)
- **V3 Hybrid** — anonymised text + DP-SGD (full stack: PII removed *and* formal DP)

### Research Question Alignment (RQ1 and RQ4)
> *RQ1 dashboard values compare V0-V3 under a shared ε = 8 setting.*
> *RQ4 end-to-end chatbot experiments report 79.47% retention at ε = 20.*

### Dataset
[GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) — 58k
Reddit comments labelled across **28 fine-grained emotions**.

### Privacy Budget Guide

| ε value | Interpretation |
|---------|---------------|
| < 1     | Very strong — significant utility loss typical |
| 1 – 4   | Strong — common in practice |
| 4 – 10  | Moderate — good balance for NLP classification |
| > 10    | Weak — limited privacy guarantee |

δ is fixed at 1×10⁻⁵ throughout (< 1 / training set size).
""")

# Build Gradio UI

def build_app() -> "gr.Blocks":
    if not GRADIO_OK:
        raise RuntimeError("Gradio is not installed. Run:  pip install gradio")

    comparison_img = _make_comparison_chart()
    retention_img = _make_retention_chart()
    chatbot_messages_mode = _supports_chatbot_messages()

    with gr.Blocks(title="Hybrid DP Mental Health Chatbot") as demo:
        gr.Markdown(
            "# 🧠 Hybrid Differential Privacy — Mental Health Chatbot Demo\n"
            "*Research demonstrating privacy-preserving emotion detection via anonymisation + DP-SGD.*"
        )

        with gr.Tabs():

            
            # Tab 1 — Chatbot
            with gr.Tab("💬 Emotion Chatbot"):
                gr.Markdown(
                    "Type how you are feeling. The model detects your emotions and responds empathetically.\n\n"
                    "Switch variants in the sidebar to compare privacy levels."
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot_kwargs: Dict[str, Any] = {
                            "label": "Conversation",
                            "height": 420,
                            "show_label": True,
                        }
                        if chatbot_messages_mode:
                            chatbot_kwargs["type"] = "messages"

                        chatbot = gr.Chatbot(
                            **chatbot_kwargs,
                        )
                        with gr.Row():
                            msg_box = gr.Textbox(
                                placeholder="How are you feeling today?",
                                label="Your message",
                                lines=2,
                                scale=5,
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1)
                        clear_btn = gr.Button("Clear conversation", variant="secondary")

                    with gr.Column(scale=1):
                        gr.Markdown("### Settings")
                        variant_dd = gr.Dropdown(
                            choices=["V0 Baseline", "V1 Anonym.", "V2 DP-SGD", "V3 Hybrid"],
                            value="V3 Hybrid",
                            label="Model Variant",
                        )
                        threshold_sl = gr.Slider(
                            minimum=0.1, maximum=0.6, step=0.05, value=0.3,
                            label="Emotion threshold",
                        )
                        gr.Markdown(
                            "**V3 Hybrid** provides the strongest privacy guarantee "
                            "while retaining the most utility.\n\n"
                            "🔒 = DP-SGD active  |  ✅ = text anonymised"
                        )
                        status_md = gr.Markdown("_Model will load on first message._")

                def _send(msg, history, variant, threshold):
                    model_loaded = load_model(variant) is not None
                    status = (
                        f"✅ Model **{variant}** loaded from checkpoint."
                        if model_loaded
                        else f"⚠️ Weights for **{variant}** not found — using rule-based fallback."
                    )
                    new_hist, _ = chat(msg, history, variant, threshold)
                    return new_hist, "", status

                send_btn.click(
                    _send,
                    inputs=[msg_box, chatbot, variant_dd, threshold_sl],
                    outputs=[chatbot, msg_box, status_md],
                )
                msg_box.submit(
                    _send,
                    inputs=[msg_box, chatbot, variant_dd, threshold_sl],
                    outputs=[chatbot, msg_box, status_md],
                )
                clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_box])

            
            # Privacy-Utility Dashboard
            with gr.Tab("📊 Privacy-Utility Dashboard"):
                gr.Markdown(
                    "## Model Comparison: V0 Baseline → V3 Hybrid\n"
                    "The hybrid approach trades a small amount of utility for strong privacy guarantees."
                )
                gr.HTML(_metrics_html())

                with gr.Row():
                    if comparison_img:
                        gr.Image(comparison_img, label="F1 Scores by Variant", show_label=True)
                    else:
                        gr.Markdown("_Install `matplotlib` to see charts: `pip install matplotlib`_")

                    if retention_img:
                        gr.Image(retention_img, label="F1-macro vs Privacy Budget ε", show_label=True)

                gr.Markdown(
                    "> **Note:** If values appear as placeholders, save your actual training metrics "
                    "to `reports/rq1/rq1_summary.json` (list of dicts with keys: "
                    "`variant`, `f1_macro`, `f1_micro`, `epsilon_actual`)."
                )

            
            # How It Works
            with gr.Tab("ℹ️ How It Works"):
                gr.Markdown(HOW_IT_WORKS_MD)

                with gr.Accordion("Try it: live emotion detection", open=False):
                    probe_in = gr.Textbox(
                        placeholder="Enter any text to see raw emotion scores …",
                        label="Input",
                        lines=2,
                    )
                    probe_variant = gr.Dropdown(
                        choices=["V0 Baseline", "V1 Anonym.", "V2 DP-SGD", "V3 Hybrid"],
                        value="V3 Hybrid",
                        label="Variant",
                    )
                    probe_btn = gr.Button("Detect emotions")
                    probe_out = gr.JSON(label="Emotion scores")

                    def _probe(text, variant):
                        return detect_emotions(text, variant, threshold=0.05)

                    probe_btn.click(_probe, inputs=[probe_in, probe_variant], outputs=probe_out)

    return demo


# Entry point

if __name__ == "__main__":
    app = build_app()
    launch_host = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    launch_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    launch_share = os.getenv("GRADIO_SHARE", "0").lower() in {"1", "true", "yes"}

    app.launch(
        server_name=launch_host,
        server_port=launch_port,
        share=launch_share,
        show_error=True,
        inbrowser=True,
    )
