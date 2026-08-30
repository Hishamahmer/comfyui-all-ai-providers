"""When each word is actually spoken, measured from the audio instead of guessed.

The moving caption styles — `highlight`, `karaoke`, `underline`, `word_by_word` — have to
know when each word lands. Until now that was an ESTIMATE: `ass_captions.estimate_words`
shares a cue's span between its words in proportion to their letter count, plus a small
bonus after punctuation. Its own docstring is honest about the limit — *"NOT
frame-accurate, and it drifts"* — and drift is exactly what a viewer notices, because the
highlight sitting one word behind the voice is more distracting than no highlight at all.

The estimate is wrong in a specific, unavoidable way: letter count is not duration.
"through" is seven letters and one beat; "AI" is two letters and two syllables spoken
slowly. Any weighting scheme is guessing at something the audio already knows.

So this node asks the audio. Whisper returns per-word timestamps, and those replace the
estimate entirely — both the word marks AND the cue boundaries, which were previously
shared out by the same proportional guess.

Two settings are load-bearing, both learned the hard way on this machine:

  * **`large-v3`, not `turbo`.** Turbo is faster and its timestamps are looser.
  * **Never pass `chunk_length_s`.** It switches transformers to the chunked algorithm,
    whose timestamps go astray at every chunk seam. Omitting it uses Whisper's sequential
    long-form decoding, which carries context across the whole clip.

A previous one-off measured this at **95.2%** — 300 of 315 sampled moments highlighting
the word actually being spoken. The estimate is nowhere near that.

**It transcribes rather than force-aligns**, and for this pipeline that is the right way
round: the narration is a text-to-speech reading of a script we wrote, so the words come
back the same, and on the rare occasion the voice says something slightly different the
captions then match what was SAID rather than what was planned.
"""

from __future__ import annotations

import json

# Cached on this machine already. `large-v3` is the accurate one; the tiny/turbo variants
# trade exactly the thing this node exists to provide.
MODELS = ["openai/whisper-large-v3", "openai/whisper-medium", "openai/whisper-small"]
LANGUAGES = ["auto", "english", "hindi", "spanish", "french", "german", "portuguese",
             "italian", "japanese", "korean", "chinese", "arabic"]


def _to_mono_16k(audio):
    """ComfyUI AUDIO -> the float32 mono 16 kHz Whisper expects."""
    import torch

    waveform = audio["waveform"]
    rate = int(audio["sample_rate"])
    if waveform.dim() == 3:                       # (batch, channels, samples)
        waveform = waveform[0]
    if waveform.dim() == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)   # down-mix, never pick one channel
    waveform = waveform.reshape(-1).float()
    if rate != 16000:
        import torchaudio
        waveform = torchaudio.functional.resample(waveform, rate, 16000)
    return waveform.cpu().numpy(), 16000


def transcribe_words(audio, model_id, language="auto", device=None):
    """[{word, start, end}, ...] for one clip. Raises with a readable reason."""
    import torch
    from transformers import pipeline

    samples, rate = _to_mono_16k(audio)
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    kwargs = {}
    if language and language != "auto":
        kwargs["generate_kwargs"] = {"language": language}

    asr = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=device,
        torch_dtype=torch.float16 if device != -1 else torch.float32,
    )
    try:
        # NO `chunk_length_s` — see the module docstring. This is the whole reason the
        # timestamps are usable.
        result = asr({"raw": samples, "sampling_rate": rate},
                     return_timestamps="word", **kwargs)
    finally:
        del asr
        if device != -1:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    words = []
    for chunk in (result.get("chunks") or []):
        stamp = chunk.get("timestamp") or (None, None)
        start, end = stamp[0], stamp[1]
        text = (chunk.get("text") or "").strip()
        if not text or start is None:
            continue
        # A final word sometimes comes back with no end. Give it a beat rather than
        # dropping it — a missing last word is the one a viewer notices.
        if end is None:
            end = float(start) + 0.3
        words.append({"word": text, "start": float(start), "end": float(end)})
    return words, (result.get("text") or "").strip()


class ArkWordTimings:
    CATEGORY = "arkennemasis/Video"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("word_timings", "transcript", "word_count", "report")
    DESCRIPTION = ("Measure when each word is actually spoken, with Whisper. Wire into "
                   "ArkVideoAssemble's `word_timings` so the moving caption styles mark "
                   "the real word instead of an estimate that drifts.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "narration": ("AUDIO", {
                    "tooltip": "The finished voice track — the same audio that goes into "
                               "the video, so the timings line up with what is heard.",
                }),
                "model": (MODELS, {
                    "tooltip": "large-v3 is the accurate one and is already cached here. "
                               "The smaller models are faster and their timestamps are "
                               "looser, which defeats the point of this node.",
                }),
                "language": (LANGUAGES, {
                    "tooltip": "Naming the language is slightly more reliable than "
                               "letting Whisper detect it on a short clip.",
                }),
            },
            "optional": {
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Off returns nothing and ArkVideoAssemble falls back to "
                               "the estimate. Whisper is a ~3 GB model load, so this is "
                               "the switch for when you do not care about drift.",
                }),
            },
        }

    def run(self, narration, model=MODELS[0], language="auto", enabled=True):
        if not enabled:
            print("[arkennemasis] word timings: disabled — captions will use the estimate.")
            return ("", "", 0, "disabled")

        try:
            words, transcript = transcribe_words(narration, model, language)
        except Exception as exc:
            # Captions that drift are a blemish; a failed run is a lost video. Fall back
            # loudly rather than taking the whole thing down for a subtitle refinement.
            print("[arkennemasis] word timings: FAILED (%s) — falling back to the "
                  "estimate, so captions may drift." % exc)
            return ("", "", 0, "failed: %s" % exc)

        if not words:
            print("[arkennemasis] word timings: Whisper returned no words — falling back "
                  "to the estimate.")
            return ("", transcript, 0, "no words returned")

        span = words[-1]["end"] - words[0]["start"]
        report = ("%d words over %.2fs | %s | first %.2fs, last ends %.2fs"
                  % (len(words), span, model.split("/")[-1],
                     words[0]["start"], words[-1]["end"]))
        print("[arkennemasis] word timings: %s" % report)
        return (json.dumps(words, ensure_ascii=False), transcript, len(words), report)


NODE_CLASS_MAPPINGS = {"ArkWordTimings": ArkWordTimings}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkWordTimings": "arkennemasis Word Timings (real caption sync, from the audio)",
}
