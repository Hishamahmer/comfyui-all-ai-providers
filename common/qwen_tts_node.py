"""Qwen3-TTS in ComfyUI: speech from text, cloned from a reference clip.

**Why this runs in a subprocess.**

Qwen3-TTS is written against `transformers==4.57.3`. This install runs 5.14.1, and eleven
other node packs depend on that, so downgrading in place is not an option — and every
published ComfyUI wrapper for this model tells you to do exactly that.

Running it in-process on 5.x was tried properly first. Four API changes were shimmed —
`check_model_inputs` became a plain decorator, `PretrainedConfig` stopped defining the
token-id attributes, `ROPE_INIT_FUNCTIONS["default"]` was deleted, and the mask helpers
renamed `input_embeds` and dropped `cache_position`. The model then loaded and ran, and
diverged inside attention, where head counts no longer matched. Chasing transformers
internals further is not something anyone should have to maintain.

So the model gets its own `transformers`, in `vendor/tts_env`, first on `sys.path` in a
child process. Same interpreter, same torch, same CUDA — only `transformers` differs, and
only there. Verified end to end: 6.30 s of cloned speech at 24 kHz on transformers 4.57.3
while the parent stayed on 5.14.1.

Everything lives under ComfyUI: models in ``ComfyUI/models/qwen-tts/``, the private
environment and the vendored package inside this node pack. Nothing points at a launcher's
private cache, so this survives being handed to someone else.

**First-time setup** (once, from the portable root)::

    python_embeded\\python.exe -m pip install -t ComfyUI\\custom_nodes\\comfyui-arkennemasis\\vendor\\tts_env ^
        --no-deps "transformers==4.57.3" "huggingface-hub>=0.34.0,<1.0" "tokenizers>=0.22.0,<=0.23.0"

    # models, into ComfyUI's own tree
    huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir ComfyUI\\models\\qwen-tts\\Qwen3-TTS-12Hz-0.6B-Base
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACK = os.path.dirname(_HERE)
_VENDOR = os.path.join(_PACK, "vendor")
_TTS_ENV = os.path.join(_VENDOR, "tts_env")
_WORKER = os.path.join(_HERE, "qwen_tts_worker.py")

# Qwen3-TTS occasionally never emits end-of-speech and generates until something stops it.
# It is a property of the sampled path, so a different seed almost always clears it —
# which is why the retry re-seeds rather than simply re-running. Three attempts because a
# second failure is already unlikely and a third is close to certain to be the text.
ATTEMPTS = 3

# ── The served worker ────────────────────────────────────────────────────────
# Loading the model costs ~18 s; speaking a seven-second line costs ~8 s. A process
# per line therefore spends most of a long run reloading the same 2.5 GB. One worker,
# kept alive and fed line by line, pays that once.
#
# Keyed on what is baked in at load time (model + environment): those cannot change
# after `from_pretrained`, so a different key retires the worker rather than reusing it.
_SERVER = {"proc": None, "key": None}
NEWLINE = chr(10)          # avoids escape-mangling through tooling


def _stop_server():
    proc = _SERVER.get("proc")
    _SERVER["proc"] = _SERVER["key"] = None
    if proc and proc.poll() is None:
        try:
            proc.stdin.write(json.dumps({"stop": True}) + NEWLINE)
            proc.stdin.flush()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def _server(key, job_path):
    """A live worker for `key`, started if there is not one already.

    The first job travels in the job FILE exactly as one-shot mode does, so the
    startup path stays identical and there is only one way for it to be wrong.
    """
    proc = _SERVER.get("proc")
    if proc is not None and proc.poll() is None and _SERVER.get("key") == key:
        return proc, False
    _stop_server()
    proc = subprocess.Popen(
        [sys.executable, _WORKER, job_path, "--serve"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1)
    _SERVER["proc"], _SERVER["key"] = proc, key
    return proc, True


def _ask(proc, job, timeout):
    """One job in, one JSON reply out. None if the worker stops answering."""
    import threading

    box = {}

    def read():
        try:
            box["line"] = proc.stdout.readline()
        except Exception as exc:
            box["error"] = exc

    if job is not None:                  # None = first job, already in the file
        proc.stdin.write(json.dumps(job) + NEWLINE)
        proc.stdin.flush()
    t = threading.Thread(target=read, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive() or not box.get("line"):
        return None                      # ran away, died, or went silent
    for piece in box["line"].splitlines():
        piece = piece.strip()
        if piece.startswith("{"):
            try:
                return json.loads(piece)
            except ValueError:
                pass
    return None


_LANGUAGES = ["Auto", "English", "Chinese", "Japanese", "Korean", "German", "French",
              "Russian", "Portuguese", "Spanish", "Italian"]


def _models_dir() -> str:
    import folder_paths
    return os.path.join(folder_paths.models_dir, "qwen-tts")


def model_choices() -> list:
    """Model folders the user actually has. Tokenizer folders are not models."""
    root = _models_dir()
    if not os.path.isdir(root):
        return ["<no models — see this node's docstring>"]
    out = [d for d in sorted(os.listdir(root))
           if os.path.isdir(os.path.join(root, d)) and "Tokenizer" not in d]
    return out or ["<no models under ComfyUI/models/qwen-tts>"]


def _write_reference(audio) -> str:
    """ComfyUI AUDIO -> a temp wav the worker can read."""
    import soundfile as sf

    waveform, sample_rate = audio["waveform"], int(audio["sample_rate"])
    array = waveform.detach().cpu().float().numpy()
    while array.ndim > 1:                       # (batch, channels, samples) -> mono
        array = array.mean(axis=0)
    handle, path = tempfile.mkstemp(suffix=".wav", prefix="ark_qwen_ref_")
    os.close(handle)
    sf.write(path, array, sample_rate)
    return path


class ArkQwenTTS:
    CATEGORY = "arkennemasis/Audio"
    FUNCTION = "run"
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "report")
    DESCRIPTION = (
        "Qwen3-TTS, running locally on the GPU with no account. Wire a clip into "
        "`reference_audio` and it speaks your text in that voice."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "What to say.",
                }),
                "model": (model_choices(), {
                    "tooltip": "A folder under ComfyUI/models/qwen-tts. Voice cloning "
                               "needs a *Base* model.",
                }),
                "language": (_LANGUAGES, {
                    "tooltip": "'Auto' lets the model decide from the text.",
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "reference_audio": ("AUDIO", {
                    "tooltip": "The voice to clone. 5-30 seconds of clean speech is "
                               "plenty. A *Base* model REQUIRES this.",
                }),
                "reference_text": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "A transcript of the reference clip. Supplying it gives a "
                               "closer clone; leaving it blank uses x-vector-only mode, "
                               "which still works.",
                }),
                "timeout_seconds": ("INT", {
                    "default": 180, "min": 60, "max": 3600,
                    "tooltip": "Per attempt, and there are 3 attempts with fresh seeds. "
                               "A 24-word line takes about 30 s including the ~2.5 GB "
                               "weight load, so 180 s is six times over — it is a "
                               "runaway detector, not a budget. It was 600 s, which "
                               "meant one looping line cost 10 minutes before failing.",
                }),
            },
        }

    def run(self, text, model, language, seed, reference_audio=None, reference_text="",
            timeout_seconds=180):
        import numpy as np
        import torch

        if not (text or "").strip():
            raise RuntimeError("ArkQwenTTS: `text` is empty.")
        if not os.path.isdir(_TTS_ENV):
            raise RuntimeError(
                f"ArkQwenTTS: the private transformers environment is missing at "
                f"{_TTS_ENV}. See this node's docstring for the one-off install command.")

        model_dir = os.path.join(_models_dir(), model)
        if not os.path.isdir(model_dir):
            raise RuntimeError(f"ArkQwenTTS: no such model folder: {model_dir}")

        # A *Base* model has no preset voices — it exists to clone one. Say so here
        # rather than let it fail 20 seconds later inside the worker.
        if reference_audio is None and "Base" in model:
            raise RuntimeError(
                f"ArkQwenTTS: '{model}' is a Base model, which only does voice cloning "
                "— wire a clip into `reference_audio`. For preset voices, download a "
                "CustomVoice model into ComfyUI/models/qwen-tts/ instead.")

        ref_path = _write_reference(reference_audio) if reference_audio is not None else None
        out_handle, out_path = tempfile.mkstemp(suffix=".wav", prefix="ark_qwen_out_")
        os.close(out_handle)
        job_handle, job_path = tempfile.mkstemp(suffix=".json", prefix="ark_qwen_job_")
        os.close(job_handle)

        job = {"env_dir": _TTS_ENV, "vendor_dir": _VENDOR, "model_dir": model_dir,
               "text": text, "language": None if language == "Auto" else language,
               "seed": int(seed), "ref_audio": ref_path,
               "ref_text": (reference_text or "").strip(), "out_path": out_path}

        # A 2-word-per-second read is the measured rate for this model; anything past
        # four times that is not speech, it is the decoder looping.
        expected = max(len(text.split()) / 2.0, 1.0)
        runaway_after = max(20.0, expected * 4.0)

        try:
            for attempt in range(1, ATTEMPTS + 1):
                # A fresh seed per attempt is the whole point of retrying: a runaway is a
                # sampling path that never emits end-of-speech, and re-running the SAME
                # seed reproduces it exactly.
                job["seed"] = (int(seed) + attempt - 1) & 0xffffffffffffffff
                with open(job_path, "w", encoding="utf-8") as fh:
                    json.dump(job, fh)

                try:
                    proc = subprocess.run([sys.executable, _WORKER, job_path],
                                          capture_output=True, text=True,
                                          timeout=int(timeout_seconds))
                except subprocess.TimeoutExpired:
                    # subprocess.run has already killed the child by the time this
                    # raises, so there is nothing to clean up but the message.
                    print("[arkennemasis] Qwen3-TTS attempt %d/%d ran away (no "
                          "end-of-speech within %ds) — retrying with a new seed"
                          % (attempt, ATTEMPTS, int(timeout_seconds)))
                    if attempt == ATTEMPTS:
                        raise RuntimeError(
                            "ArkQwenTTS: generation ran away on all %d attempts. The "
                            "model never emitted end-of-speech for this line: %r. "
                            "Rephrase it, or raise timeout_seconds if the line really "
                            "is long." % (ATTEMPTS, text[:160]))
                    continue

                reply = {}
                for line in proc.stdout.splitlines():
                    if line.strip().startswith("{"):
                        reply = json.loads(line)
                if not reply.get("ok"):
                    # A real fault (missing model, bad env) repeats identically, so it
                    # is raised at once rather than retried three times.
                    detail = reply.get("error") or (proc.stderr or "")[-1200:]
                    raise RuntimeError(f"ArkQwenTTS failed: {detail}")

                import soundfile as sf
                wav, sample_rate = sf.read(out_path, dtype="float32", always_2d=False)
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                seconds = len(wav) / float(sample_rate or 1)

                if seconds > runaway_after and attempt < ATTEMPTS:
                    # Finished inside the timeout but produced far more audio than the
                    # text can account for — the same looping, caught by its output.
                    print("[arkennemasis] Qwen3-TTS attempt %d/%d produced %.1fs for "
                          "%d words (expected ~%.1fs) — retrying with a new seed"
                          % (attempt, ATTEMPTS, seconds, len(text.split()), expected))
                    continue

                how = ("voice clone" if ref_path else "model voice")
                report = (f"Qwen3-TTS {model} | {how} | {seconds:.2f}s @ {sample_rate} Hz "
                          f"| {reply.get('device')} | "
                          f"transformers {reply.get('transformers')}"
                          + (f" | attempt {attempt}" if attempt > 1 else ""))
                print(f"[arkennemasis] {report}")
                audio = {"waveform": torch.from_numpy(np.asarray(wav)[None, None, :]),
                         "sample_rate": int(sample_rate)}
                return (audio, report)
        finally:
            for path in (ref_path, out_path, job_path):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass


NODE_CLASS_MAPPINGS = {
    "ArkQwenTTS": ArkQwenTTS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkQwenTTS": "arkennemasis Qwen3-TTS (voice clone)",
}
