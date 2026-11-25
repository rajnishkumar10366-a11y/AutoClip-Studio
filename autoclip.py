"""
AutoClip Studio — single-file AI short-form clip cutter
Run: streamlit run autoclip.py

Requirements:
  pip install streamlit yt-dlp faster-whisper requests
  Install FFmpeg and make sure ffmpeg + ffprobe are on PATH.
  Optional: install Ollama and pull a model, e.g. `ollama pull llama3.2`
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path

import requests
import streamlit as st

APP_TITLE = "AutoClip Studio"
OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.2"

st.set_page_config(page_title=APP_TITLE, page_icon="✂️", layout="wide")


def run_cmd(args, timeout=None):
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout or "Command failed")[-4000:])
    return p.stdout.strip()


def ffmpeg_ready():
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def probe_duration(path):
    out = run_cmd([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ])
    return float(out)


def get_video_from_youtube(url, outdir):
    template = str(Path(outdir) / "source.%(ext)s")
    run_cmd([
        "yt-dlp",
        "--no-playlist",
        "--concurrent-fragments", "8",
        "--retries", "10",
        "--fragment-retries", "10",
        "--socket-timeout", "20",
        "-f", "bv*[height<=1080]+ba/b[height<=1080]/b[height<=1080]",
        "-S", "res:1080,fps,codec:h264",
        "--merge-output-format", "mp4",
        "-o", template,
        url
    ], timeout=3600)
    candidates = list(Path(outdir).glob("source.*"))
    candidates = [p for p in candidates if p.suffix.lower() in
                  (".mp4", ".mkv", ".webm", ".mov")]
    if not candidates:
        raise RuntimeError("YouTube download did not produce a video file.")
    return str(max(candidates, key=lambda p: p.stat().st_size))


def transcribe_video(path, model_size="tiny", language=None, audio_path=None):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError("faster-whisper is missing. Run: pip install faster-whisper")

    # Decode only the audio track once. Whisper does not need to decode video frames.
    if audio_path is None:
        audio_path = str(Path(path).with_name("autoclip_audio_16k.wav"))
    run_cmd([
        "ffmpeg", "-y", "-i", path, "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", audio_path
    ], timeout=1800)
    # Use NVIDIA CUDA automatically when CTranslate2 can see a compatible GPU.
    # Otherwise use the fast int8 CPU path.
    try:
        import ctranslate2
        has_cuda = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        has_cuda = False
    device = "cuda" if has_cuda else "cpu"
    compute_type = "float16" if has_cuda else "int8"
    try:
        model = WhisperModel(
            model_size, device=device, compute_type=compute_type,
            cpu_threads=max(2, (os.cpu_count() or 4) - 1),
            num_workers=1
        )
    except TypeError as exc:
        if "metadata_errors" in str(exc) or "unexpected keyword argument" in str(exc):
            raise RuntimeError(
                "A dependency version conflict is preventing Whisper from loading. "
                "In CMD run: python -m pip install --upgrade --force-reinstall "
                "huggingface-hub httpx requests, then restart AutoClip."
            ) from exc
        raise
    lang = None if language == "Auto Detect" else (
        "hi" if language == "Hindi" else "en" if language == "English" else None
    )
    # Batched inference is substantially faster on supported NVIDIA GPUs.
    transcribe_args = dict(
        language=lang, beam_size=1, best_of=1,
        word_timestamps=True, vad_filter=True,
        condition_on_previous_text=False
    )
    if has_cuda:
        try:
            from faster_whisper import BatchedInferencePipeline
            pipeline = BatchedInferencePipeline(model=model)
            segments, info = pipeline.transcribe(audio_path, batch_size=8, **transcribe_args)
        except Exception:
            segments, info = model.transcribe(audio_path, **transcribe_args)
    else:
        segments, info = model.transcribe(audio_path, **transcribe_args)
    rows = []
    for seg in segments:
        words = []
        if seg.words:
            for w in seg.words:
                words.append({
                    "start": float(w.start), "end": float(w.end),
                    "word": w.word.strip()
                })
        rows.append({
            "start": float(seg.start), "end": float(seg.end),
            "text": seg.text.strip(), "words": words
        })
    return rows, getattr(info, "language", "unknown")


def build_candidates(segments, duration, clip_len):
    """
    Build overlapping, transcript-backed windows across the whole video.
    Overlap gives the ranker a better chance of catching a hook that starts
    just before/after an arbitrary timestamp.
    """
    if not segments:
        return []

    step = max(8, clip_len * 0.45)
    last_start = max(0.0, duration - clip_len)
    starts = []
    t = 0.0
    while t <= last_start:
        starts.append(round(t, 1))
        t += step
    if not starts or starts[-1] < last_start - 1:
        starts.append(round(last_start, 1))

    candidates = []
    for start in starts:
        end = min(duration, start + clip_len)
        overlapping = [
            seg for seg in segments
            if seg["end"] > start and seg["start"] < end
        ]
        text = " ".join(seg["text"] for seg in overlapping).strip()
        if len(text.split()) < 8:
            continue
        candidates.append({
            "id": len(candidates),
            "start": start,
            "end": round(end, 1),
            "text": text[:1400]
        })
    return candidates


def _parse_ollama_json(content):
    """Accept clean JSON and common fenced/wrapped JSON responses."""
    content = (content or "").strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            raise RuntimeError("Ollama did not return a JSON object.")
        return json.loads(match.group(0))


def _ask_ollama_for_picks(candidates, model, limit):
    compact = [{
        "id": c["id"],
        "time": f'{int(c["start"]//60):02d}:{int(c["start"]%60):02d}',
        "transcript": c["text"][:1000]
    } for c in candidates]

    prompt = (
        "You are an expert short-form video editor. Choose the most compelling "
        "self-contained moments for Shorts/Reels. Score each candidate 1-10. "
        "Prioritize: an immediate hook in the first seconds, surprising or useful "
        "insight, strong opinion, emotional reaction, clear mini-story, specific "
        "example, actionable advice, or a memorable punchline. Penalize: greetings, "
        "long setup, repetition, vague talk, dead air, incomplete thoughts, and "
        "moments that need lots of missing context. Prefer different topics/times. "
        "Do not judge by keyword presence alone. IDs must be copied exactly from "
        "the input. Return ONLY JSON with this exact shape: "
        '{"picks":[{"id":0,"score":9,"reason":"short specific reason"}]}. '
        f"Return at most {limit} picks, best first. If none are usable, return "
        '{"picks":[]}.\n\nCANDIDATES:\n'
        + json.dumps(compact, ensure_ascii=False)
    )

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "num_ctx": 8192,
                "num_predict": 900
            },
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=360
    )
    response.raise_for_status()
    body = response.json()
    content = body.get("message", {}).get("content", "")
    data = _parse_ollama_json(content)
    if isinstance(data, list):
        data = {"picks": data}
    if not isinstance(data, dict):
        raise RuntimeError("Ollama response had an unexpected JSON shape.")

    valid = {int(c["id"]): c for c in candidates}
    picks = []
    for item in data.get("picks", []):
        if not isinstance(item, dict):
            continue
        try:
            cid = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        if cid not in valid or any(p["id"] == cid for p in picks):
            continue
        pick = dict(valid[cid])
        try:
            pick["score"] = max(1, min(10, int(float(item.get("score", 7)))))
        except (TypeError, ValueError):
            pick["score"] = 7
        pick["reason"] = str(item.get("reason") or "AI-selected engaging moment")[:220]
        picks.append(pick)
    return picks


def ollama_rank(candidates, model):
    """
    Rank candidates in small batches so a 30-minute transcript does not
    overwhelm the local model/context window. Then rerank the shortlist.
    """
    if not candidates:
        return []

    batch_size = 10
    shortlist = []
    for offset in range(0, len(candidates), batch_size):
        batch = candidates[offset:offset + batch_size]
        batch_picks = _ask_ollama_for_picks(batch, model, limit=3)
        shortlist.extend(batch_picks)

    # Keep the best candidates from each batch, then make one global selection.
    if not shortlist:
        return []
    shortlist = sorted(shortlist, key=lambda x: x.get("score", 0), reverse=True)[:24]
    final_picks = _ask_ollama_for_picks(shortlist, model, limit=8)
    if not final_picks:
        final_picks = shortlist

    # Enforce time diversity so clips do not cover almost the same moment.
    final_picks = sorted(final_picks, key=lambda x: x.get("score", 0), reverse=True)
    selected = []
    for pick in final_picks:
        overlaps = any(
            min(pick["end"], old["end"]) - max(pick["start"], old["start"])
            > 0.35 * min(pick["end"] - pick["start"], old["end"] - old["start"])
            for old in selected
        )
        if not overlaps:
            selected.append(pick)
        if len(selected) >= 8:
            break
    return selected


def fallback_rank(candidates):
    """
    Better deterministic fallback: combines hook, specificity, emotion,
    useful advice, completeness, and speech-density signals.
    It is a backup, not a substitute for semantic AI ranking.
    """
    hook_terms = (
        "why", "how", "secret", "mistake", "never", "best", "because",
        "important", "actually", "truth", "problem", "reason", "learn",
        "क्यों", "कैसे", "गलती", "सच", "जरूरी", "लेकिन", "इसलिए"
    )
    emotion_terms = (
        "love", "hate", "fear", "crazy", "shocking", "amazing", "worst",
        "best", "fail", "success", "angry", "happy", "डर", "कमाल", "गलत"
    )
    advice_terms = (
        "you should", "you need to", "remember", "tip", "step", "instead",
        "try this", "the lesson", "आपको", "याद रखो", "ध्यान"
    )
    filler_terms = (
        "welcome back", "hello everyone", "like and subscribe",
        "in today's video", "so yeah", "you know what i mean"
    )

    scored = []
    for c in candidates:
        text = c["text"].lower()
        words = re.findall(r"\w+", text, flags=re.UNICODE)
        word_count = len(words)
        hook_count = sum(1 for term in hook_terms if term in text)
        emotion_count = sum(1 for term in emotion_terms if term in text)
        advice_count = sum(1 for term in advice_terms if term in text)
        number_bonus = 1 if re.search(r"\b\d+(?:\.\d+)?%?\b", text) else 0
        question_bonus = 1 if "?" in text or any(q in text for q in ("why ", "how ", "क्यों", "कैसे")) else 0
        filler_penalty = sum(1 for term in filler_terms if term in text)
        # Reward enough spoken content, but avoid simply favoring long transcripts.
        density_score = min(word_count / 45, 2.0)
        completeness_bonus = 0.7 if text.rstrip().endswith((".", "!", "?", "।")) else 0
        raw = (
            3.0 + density_score + min(hook_count, 3) * 0.65
            + min(emotion_count, 2) * 0.55
            + min(advice_count, 2) * 0.6
            + number_bonus * 0.35 + question_bonus * 0.4
            + completeness_bonus - filler_penalty * 1.0
        )
        score = max(1, min(10, round(raw)))
        scored.append(dict(
            c, score=score,
            reason="Backup score: hook/specificity/usefulness signals; review this selection"
        ))

    scored.sort(key=lambda x: x["score"], reverse=True)
    selected = []
    for pick in scored:
        overlaps = any(
            min(pick["end"], old["end"]) - max(pick["start"], old["start"])
            > 0.35 * min(pick["end"] - pick["start"], old["end"] - old["start"])
            for old in selected
        )
        if not overlaps:
            selected.append(pick)
        if len(selected) >= 8:
            break
    return selected

def ass_time(seconds):
    cs = max(0, int(round(seconds * 100)))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, centi = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{centi:02d}"


def ass_escape(s):
    return s.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def make_ass(segments, clip_start, clip_end, output_path, width, height,
             font_family="Arial", caption_position="Bottom", caption_style="Hormozi Yellow"):
    presets = {
        "Hormozi Yellow": {"primary": "&H00FFFFFF", "highlight": "&H0000D7FF", "outline": "&H00000000", "back": "&H90000000", "bold": -1, "italic": 0, "border": 1, "outline_scale": 0.065, "shadow_scale": 0.025, "uppercase": True},
        "Karaoke Cyan": {"primary": "&H00FFFFFF", "highlight": "&H00FFFF00", "outline": "&H00101010", "back": "&H90000000", "bold": -1, "italic": 0, "border": 1, "outline_scale": 0.07, "shadow_scale": 0.03, "uppercase": True},
        "Classic Subtitles": {"primary": "&H00FFFFFF", "highlight": "&H00FFFFFF", "outline": "&H00000000", "back": "&H80000000", "bold": 0, "italic": 0, "border": 1, "outline_scale": 0.045, "shadow_scale": 0.02, "uppercase": False},
        "Bold White": {"primary": "&H00FFFFFF", "highlight": "&H00FFFFFF", "outline": "&H00000000", "back": "&H00000000", "bold": -1, "italic": 0, "border": 1, "outline_scale": 0.09, "shadow_scale": 0.035, "uppercase": True},
        "Neon Green": {"primary": "&H0000FF7F", "highlight": "&H0000FFFF", "outline": "&H00101010", "back": "&H90000000", "bold": -1, "italic": 0, "border": 1, "outline_scale": 0.075, "shadow_scale": 0.03, "uppercase": True},
        "Red Highlight": {"primary": "&H00FFFFFF", "highlight": "&H000000FF", "outline": "&H00000000", "back": "&H90000000", "bold": -1, "italic": 0, "border": 1, "outline_scale": 0.07, "shadow_scale": 0.03, "uppercase": True},
        "Purple Pop": {"primary": "&H00FFFFFF", "highlight": "&H00FF80FF", "outline": "&H00201030", "back": "&H90000000", "bold": -1, "italic": 0, "border": 1, "outline_scale": 0.075, "shadow_scale": 0.03, "uppercase": True},
        "Minimal Clean": {"primary": "&H00FFFFFF", "highlight": "&H00FFFFFF", "outline": "&H00000000", "back": "&H70000000", "bold": 0, "italic": 0, "border": 1, "outline_scale": 0.025, "shadow_scale": 0.01, "uppercase": False},
        "Social Media Box": {"primary": "&H00000000", "highlight": "&H00000000", "outline": "&H00FFFFFF", "back": "&H00FFFFFF", "bold": -1, "italic": 0, "border": 3, "outline_scale": 0.08, "shadow_scale": 0.0, "uppercase": True},
        "Italic Editorial": {"primary": "&H00FFFFFF", "highlight": "&H0000D7FF", "outline": "&H00000000", "back": "&H90000000", "bold": 0, "italic": -1, "border": 1, "outline_scale": 0.05, "shadow_scale": 0.02, "uppercase": False},
        "No Highlight": {"primary": "&H00FFFFFF", "highlight": "&H00FFFFFF", "outline": "&H00000000", "back": "&H90000000", "bold": -1, "italic": 0, "border": 1, "outline_scale": 0.065, "shadow_scale": 0.025, "uppercase": True},
    }
    preset = presets.get(caption_style, presets["Hormozi Yellow"])
    # Ratio-aware ASS canvas keeps text size and safe margins consistent.
    font_size = max(34, int(min(width * 0.065, height * 0.055)))
    margin_l = int(width * 0.06)
    margin_r = margin_l
    if caption_position == "Center":
        alignment, margin_v = 5, 0
    elif caption_position == "Top":
        alignment, margin_v = 8, int(height * 0.12)
    else:
        alignment, margin_v = 2, int(height * 0.16)
    outline = max(2, int(font_size * preset["outline_scale"]))
    shadow = max(0, int(font_size * preset["shadow_scale"]))
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_family},{font_size},{preset["primary"]},{preset["highlight"]},{preset["outline"]},{preset["back"]},{preset["bold"]},{preset["italic"]},{preset["border"]},{outline},{shadow},{alignment},{margin_l},{margin_r},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    all_words = []
    for seg in segments:
        for w in seg.get("words", []):
            if w["end"] > clip_start and w["start"] < clip_end:
                all_words.append(w)
    if all_words:
        # Display short groups of up to 5 words; highlight each word in sequence.
        for i in range(0, len(all_words), 5):
            group = all_words[i:i+5]
            for j, w in enumerate(group):
                start = max(clip_start, w["start"])
                end = min(clip_end, w["end"])
                if end <= start:
                    continue
                parts = []
                for k, item in enumerate(group):
                    raw_word = item["word"].upper() if preset["uppercase"] else item["word"]
                    word = ass_escape(raw_word)
                    if k == j and caption_style != "No Highlight" and preset["highlight"] != preset["primary"]:
                        parts.append("{\\c" + preset["highlight"] + "}" + word + "{\\c" + preset["primary"] + "}")
                    else:
                        parts.append(word)
                txt = " ".join(parts)
                events.append(
                    f"Dialogue: 0,{ass_time(start-clip_start)},{ass_time(end-clip_start)},"
                    f"Default,,0,0,0,,{txt}"
                )
    else:
        for seg in segments:
            start, end = max(clip_start, seg["start"]), min(clip_end, seg["end"])
            if end > start:
                raw_text = seg["text"].upper() if preset["uppercase"] else seg["text"]
                txt = ass_escape(raw_text)
                events.append(
                    f"Dialogue: 0,{ass_time(start-clip_start)},{ass_time(end-clip_start)},"
                    f"Default,,0,0,0,,{txt}"
                )
    Path(output_path).write_text(header + "\n".join(events), encoding="utf-8")


def render_clip(source, pick, segments, outpath, captions=True, ratio="9:16",
                font_family="Arial", caption_position="Bottom", caption_style="Hormozi Yellow"):
    start, end = float(pick["start"]), float(pick["end"])
    duration = end - start
    dimensions = {
        "9:16 (Shorts/Reels/TikTok)": (1080, 1920),
        "1:1 (Square)": (1080, 1080),
        "16:9 (YouTube)": (1920, 1080),
        "4:5 (Instagram portrait)": (1080, 1350),
    }
    width, height = dimensions.get(ratio, (1080, 1920))
    vf = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
    if captions:
        ass_path = str(Path(outpath).with_suffix(".ass"))
        make_ass(segments, start, end, ass_path, width, height,
                 font_family=font_family, caption_position=caption_position,
                 caption_style=caption_style)
        # FFmpeg subtitles filter requires escaped path on Windows.
        escaped = ass_path.replace("\\", "/").replace(":", r"\:")
        vf += f",subtitles='{escaped}'"
    run_cmd([
        "ffmpeg", "-y", "-ss", str(start), "-i", source, "-t", str(duration),
        "-vf", vf, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-threads", "0", "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart", outpath
    ], timeout=1800)


def main():
    st.title("✂️ AutoClip Studio")
    st.caption("Turn a long video into AI-selected Shorts, Reels and TikToks. Downloads are capped at 1080p; no 2K/4K source downloads.")
    if not ffmpeg_ready():
        st.error("FFmpeg/ffprobe not found. Install FFmpeg and add its bin folder to Windows PATH, then restart the terminal.")
        st.code("winget install Gyan.FFmpeg")
        st.stop()

    # Keep previews contained and centered; never stretch vertical clips to the full page width.
    st.markdown("""<style>
    [data-testid="stVideo"] { max-width: 760px; margin-left: auto; margin-right: auto; }
    [data-testid="stVideo"] video { max-height: 72vh; width: 100%; object-fit: contain; background: #111; }
    </style>""", unsafe_allow_html=True)

    with st.sidebar:
        st.header("Clip settings")
        clip_len = st.slider("Target clip length (seconds)", 15, 90, 45)
        auto_count = st.toggle("Automatically choose clip count from video length", value=True)
        max_clips = st.slider("Maximum number of clips", 1, 10, 6, disabled=auto_count)
        fast_mode = st.toggle("Fast mode (skip slow Ollama ranking)", value=True,
                              help="Uses the built-in transcript ranker instead of sending many requests to local Ollama. This saves time, but AI moment selection may be less accurate.")
        aspect_ratio = st.selectbox(
            "Video aspect ratio",
            ["9:16 (Shorts/Reels/TikTok)", "1:1 (Square)",
             "16:9 (YouTube)", "4:5 (Instagram portrait)"],
            index=0
        )
        captions_default = st.toggle("Captions ON by default", value=True)
        caption_style = st.selectbox(
            "Caption style",
            ["Hormozi Yellow", "Karaoke Cyan", "Classic Subtitles", "Bold White",
             "Neon Green", "Red Highlight", "Purple Pop", "Minimal Clean",
             "Social Media Box", "Italic Editorial", "No Highlight"],
            index=0, help="Choose a complete caption look. Rebuild clips to apply changes.")
        font_family = st.selectbox(
            "Caption font family",
            ["Arial", "Montserrat", "Poppins", "Roboto", "Anton", "Bebas Neue", "Impact", "DejaVu Sans"],
            index=0,
            help="Fonts must be installed on Windows for FFmpeg to render them. Arial is the safest default."
        )
        caption_position = st.selectbox(
            "Caption position", ["Bottom", "Center", "Top"], index=0,
            help="Bottom keeps captions in the lower safe area; Center and Top are alternatives."
        )
        language = st.selectbox("Caption/transcription language",
                                ["Auto Detect", "English", "Hindi", "Hinglish"])
        model_size = st.selectbox(
            "Transcription speed / accuracy",
            ["tiny (fastest)", "base (balanced)", "small (slower, more accurate)"],
            index=0,
            help="For long videos, Tiny is fastest. Base/Small may improve accuracy but take longer."
        )
        model_size = model_size.split(" ")[0]
        ollama_model = st.text_input("Ollama model", DEFAULT_MODEL)

    youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    uploaded = st.file_uploader("Or upload a video", type=["mp4", "mov", "mkv", "webm"])
    st.info("YouTube downloads are capped at 1080p maximum (no 2K/4K). Downloads use parallel fragments where supported. Only process videos you own or have permission to edit.")

    if "clips" not in st.session_state:
        st.session_state.clips = []
    if "workdir" not in st.session_state:
        st.session_state.workdir = None
    if "source" not in st.session_state:
        st.session_state.source = None
    if "segments" not in st.session_state:
        st.session_state.segments = []

    if st.button("🚀 Find viral moments and create clips", type="primary", use_container_width=True):
        if not youtube_url.strip() and uploaded is None:
            st.warning("Paste a YouTube URL or upload a video first.")
            st.stop()
        workdir = tempfile.mkdtemp(prefix="autoclip_")
        st.session_state.workdir = workdir
        try:
            started_at = time.perf_counter()
            progress = st.progress(0, text="Starting AutoClip...")
            stage_line = st.empty()
            eta_line = st.empty()
            def show_progress(percent, label, eta_seconds=None):
                elapsed = time.perf_counter() - started_at
                progress.progress(min(100, max(0, int(percent))), text=f"{int(percent)}% · {label}")
                stage_line.caption(f"Elapsed: {int(elapsed//60)}m {int(elapsed%60)}s")
                if eta_seconds is None:
                    eta_line.caption("Estimated time remaining: calculating...")
                else:
                    eta_line.caption(f"Estimated time remaining: about {int(max(0, eta_seconds)//60)}m {int(max(0, eta_seconds)%60)}s")

            show_progress(3, "Preparing video", 0)
            if uploaded is not None:
                source = str(Path(workdir) / uploaded.name)
                with open(source, "wb") as f:
                    f.write(uploaded.getbuffer())
            else:
                show_progress(5, "Downloading video", None)
                source = get_video_from_youtube(youtube_url.strip(), workdir)
            show_progress(15, "Checking video and preparing audio", None)
            duration = probe_duration(source)
            if duration < 8:
                raise RuntimeError("Video is too short to create useful clips.")

            show_progress(20, f"Transcribing with Whisper {model_size}", None)
            transcribe_started = time.perf_counter()
            segments, detected_lang = transcribe_video(source, model_size, language)
            transcribe_elapsed = time.perf_counter() - transcribe_started
            st.session_state.source = source
            st.session_state.segments = segments
            if not segments:
                raise RuntimeError("No speech was detected. This first version needs spoken audio to find highlights.")

            show_progress(55, "Finding and scoring highlight moments", None)
            candidates = build_candidates(segments, duration, clip_len)
            if not candidates:
                raise RuntimeError("Could not create candidate moments from the transcript.")

            if auto_count:
                chosen_count = max(3, min(10, round(duration / 300)))
            else:
                chosen_count = max_clips
            if fast_mode:
                picks = fallback_rank(candidates)
                ranking_note = "Fast mode: built-in transcript ranking used; local Ollama was skipped."
            else:
                try:
                    picks = ollama_rank(candidates, ollama_model.strip() or DEFAULT_MODEL)
                    if not picks:
                        raise RuntimeError("Ollama returned no usable picks after batch ranking.")
                    ranking_note = f"Ranked by Ollama AI across {len(candidates)} candidate moments."
                except Exception as e:
                    picks = fallback_rank(candidates)
                    ranking_note = (f"Ollama ranking failed; using transcript-based backup scoring. Reason: {str(e)[:220]}")

            picks = picks[:chosen_count]
            st.caption(ranking_note)
            results = []
            render_started = time.perf_counter()
            for idx, pick in enumerate(picks, 1):
                # Update ETA from measured rendering speed after the first clip.
                if idx == 1:
                    eta = None
                else:
                    avg_render = (time.perf_counter() - render_started) / (idx - 1)
                    eta = avg_render * (len(picks) - idx + 1)
                pct = 60 + int(38 * (idx - 1) / max(1, len(picks)))
                show_progress(pct, f"Rendering clip {idx}/{len(picks)}", eta)
                out = str(Path(workdir) / f"viral_clip_{idx:02d}.mp4")
                render_clip(source, pick, segments, out,
                            captions=captions_default, ratio=aspect_ratio,
                            font_family=font_family, caption_position=caption_position,
                            caption_style=caption_style)
                results.append({
                    "path": out, "start": pick["start"], "end": pick["end"],
                    "reason": pick.get("reason", ""), "score": pick.get("score", 0),
                    "captions": captions_default, "ratio": aspect_ratio,
                    "font_family": font_family, "caption_position": caption_position,
                    "caption_style": caption_style
                })
            st.session_state.clips = results
            total_elapsed = time.perf_counter() - started_at
            progress.progress(100, text="100% · All clips completed")
            stage_line.caption(f"Total processing time: {int(total_elapsed//60)}m {int(total_elapsed%60)}s")
            eta_line.success(f"Completed {len(results)} clips successfully!")
        except Exception as e:
            st.error(f"Could not finish: {e}")
            st.caption("Tip: Fast mode skips Ollama to reduce processing time. For stronger semantic selection, turn Fast mode off; this can take longer. CUDA acceleration is used only when a compatible NVIDIA GPU is available.")

    if st.session_state.clips:
        st.header("Your clips")
        st.caption("Previews are contained to preserve the original aspect ratio. Use the sidebar font and position settings, then rebuild all clips to apply them consistently.")
        if st.button("Apply current caption settings to ALL clips", type="primary", use_container_width=True):
            try:
                progress = st.progress(0, text="Rebuilding clips with the selected caption settings...")
                total = len(st.session_state.clips)
                for idx, item in enumerate(st.session_state.clips, 1):
                    render_clip(
                        st.session_state.source,
                        {"start": item["start"], "end": item["end"]},
                        st.session_state.segments, item["path"],
                        captions=item["captions"], ratio=item.get("ratio", aspect_ratio),
                        font_family=font_family, caption_position=caption_position,
                        caption_style=caption_style
                    )
                    item["font_family"] = font_family
                    item["caption_position"] = caption_position
                    item["caption_style"] = caption_style
                    progress.progress(idx / total, text=f"Updated {idx}/{total} clips")
                st.success("All clips rebuilt with the selected font and caption position.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not rebuild clips: {e}")
        for i, item in enumerate(st.session_state.clips):
            with st.expander(f"Clip {i+1} · {item['start']:.1f}s–{item['end']:.1f}s · score {item['score']}/10", expanded=False):
                st.caption(f"Format: {item.get('ratio', '9:16 (Shorts/Reels/TikTok)')}")
                st.write(item["reason"])
                preview_cols = st.columns([1, 2, 1])
                with preview_cols[1]:
                    st.video(item["path"], width=360)
                new_caption = st.toggle(
                    f"Captions on this clip", value=item["captions"],
                    key=f"caption_{i}"
                )
                if new_caption != item["captions"]:
                    if st.button("Apply setting and regenerate clip", key=f"regen_{i}"):
                        try:
                            render_clip(
                                st.session_state.source,
                                {"start": item["start"], "end": item["end"]},
                                st.session_state.segments,
                                item["path"],
                                captions=new_caption,
                                ratio=item.get("ratio", "9:16 (Shorts/Reels/TikTok)"),
                                font_family=item.get("font_family", font_family),
                                caption_position=item.get("caption_position", caption_position),
                                caption_style=item.get("caption_style", caption_style)
                            )
                            st.session_state.clips[i]["captions"] = new_caption
                            st.session_state.clips[i]["caption_style"] = caption_style
                            st.success("Clip regenerated with your caption setting. Refreshing preview...")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Could not regenerate clip: {e}")
                with open(item["path"], "rb") as f:
                    st.download_button(
                        "⬇️ Download MP4", data=f.read(),
                        file_name=f"viral_clip_{i+1:02d}.mp4",
                        mime="video/mp4", key=f"download_{i}"
                    )
        if len(st.session_state.clips) > 1:
            zip_path = str(Path(st.session_state.workdir) / "all_clips.zip")
            import zipfile
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                for i, item in enumerate(st.session_state.clips, 1):
                    z.write(item["path"], f"viral_clip_{i:02d}.mp4")
            with open(zip_path, "rb") as f:
                st.download_button("📦 Download all clips (ZIP)", f.read(),
                                   file_name="autoclip_results.zip",
                                   mime="application/zip", use_container_width=True)

    with st.expander("Setup instructions"):
        st.markdown("""
**Windows setup**

1. Install Python 3.10 or newer.
2. Install FFmpeg: `winget install Gyan.FFmpeg`
3. Install Python packages:
   ```powershell
   pip install streamlit yt-dlp faster-whisper requests
   ```
4. Install Ollama, then run:
   ```powershell
   ollama pull llama3.2
   ollama serve
   ```
5. Start this app:
   ```powershell
   streamlit run autoclip.py
   ```

The first Whisper run downloads its speech model and may take a while. Processing speed depends on CPU/GPU; Fast mode skips Ollama ranking. Auto clip count targets about one clip per five minutes of source video, capped at 10 clips (30 minutes = about 6 clips).
        """)


if __name__ == "__main__":
    main()
