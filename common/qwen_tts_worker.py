"""Run Qwen3-TTS in a subprocess that sees transformers 4.57.3 instead of 5.x.

Qwen3-TTS is written against transformers 4.57.3. This ComfyUI install runs 5.14.1, and
eleven other node packs depend on that, so downgrading in place is not an option. Shimming
5.x back to the 4.x surface got the model loading but then diverged inside attention —
porting transformers internals is not a maintainable answer.

So the model runs in its own process with a private `transformers` first on `sys.path`.
Same interpreter, same torch, same CUDA — only `transformers` differs, and only here.
Nothing about the main environment changes, which is why this survives being shared.

Invoked two ways::

    python qwen_tts_worker.py <job.json>            one line, then exit
    python qwen_tts_worker.py <job.json> --serve    stay up, read more jobs on stdin

`job.json` carries model_dir, text, language, an optional reference wav path and its
transcript, and where to write the result. The reply is one JSON line on stdout.

**--serve exists because loading the model costs more than using it.** Weights are about
2.5 GB and take ~18 s to load; a seven-second line then takes ~8 s to speak. Spawning a
process per line therefore spends most of a run loading the same weights over and over —
measured at 26 s a shot, which across a 47-shot film is 20 minutes, most
of it wasted. Served, the model is loaded once and each further line costs only what it
costs to say.

The protocol is deliberately dull: one JSON job per line in, one JSON reply per line out,
in order. No framing, no length prefixes — a line-oriented pipe is the one thing that
behaves identically on Windows and POSIX, and the parent can always fall back to
one-shot mode if anything about the pipe misbehaves.
"""

from __future__ import annotations

import json
import os
import sys


def speak(model, job, sf, torch):
    """One line of speech -> a wav on disk. Everything reusable is already loaded."""
    torch.manual_seed(int(job.get("seed") or 0))
    language = job.get("language") or None
    ref_path = job.get("ref_audio")
    if ref_path:
        import librosa
        wav, sr = librosa.load(ref_path, sr=None, mono=True)
        ref_text = (job.get("ref_text") or "").strip() or None
        wavs, out_sr = model.generate_voice_clone(
            text=job["text"], language=language, ref_audio=(wav, sr),
            ref_text=ref_text, x_vector_only_mode=ref_text is None)
    else:
        wavs, out_sr = model.generate_custom_voice(
            text=job["text"], language=language, speaker=job.get("speaker") or None)
    audio = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
    sf.write(job["out_path"], audio, out_sr)
    return int(out_sr), int(len(audio))


def main() -> int:
    job = json.loads(open(sys.argv[1], "r", encoding="utf-8").read())
    serve = "--serve" in sys.argv[2:]

    # The private transformers must win over the one in site-packages. Prepending to
    # sys.path is not enough on its own — anything already imported would keep the old
    # module — but this process has imported nothing yet.
    env_dir = job["env_dir"]
    if env_dir and os.path.isdir(env_dir):
        sys.path.insert(0, env_dir)

    vendor = job["vendor_dir"]
    if vendor not in sys.path:
        sys.path.insert(1, vendor)

    import transformers

    if not transformers.__version__.startswith("4."):
        raise RuntimeError(
            f"worker loaded transformers {transformers.__version__} from "
            f"{transformers.__file__}; expected the isolated 4.57.3. Check env_dir.")

    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    # A silent fall back to CPU is the worst outcome here: it still produces correct
    # audio, just ~40x slower, and reads exactly like a hang. Refuse it instead, unless
    # the caller has explicitly allowed it.
    if not torch.cuda.is_available():
        if not job.get("allow_cpu"):
            raise RuntimeError(
                "CUDA is not available inside the TTS worker, and CPU inference would "
                "take minutes per line. torch %s from %s. Check that vendor/tts_env "
                "shadows only transformers, not torch." % (torch.__version__,
                                                           torch.__file__))
        print("[worker] WARNING: no CUDA, running on CPU — expect minutes per line",
              file=sys.stderr)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen3TTSModel.from_pretrained(job["model_dir"], dtype=dtype,
                                          device_map=device)

    def reply(out_sr, samples):
        print(json.dumps({"ok": True, "sample_rate": out_sr, "samples": samples,
                          "device": device,
                          "gpu": (torch.cuda.get_device_name(0)
                                  if device == "cuda" else None),
                          "transformers": transformers.__version__}), flush=True)

    out_sr, samples = speak(model, job, sf, torch)
    reply(out_sr, samples)
    if not serve:
        return 0

    # Served: keep the weights and answer until stdin closes. Each further line costs
    # only synthesis. A malformed or failing job replies with ok=False and the loop
    # continues — one bad line must not take the worker down mid-run.
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            nxt = json.loads(line)
        except ValueError as exc:
            print(json.dumps({"ok": False, "error": "bad job line: %s" % exc}),
                  flush=True)
            continue
        if nxt.get("stop"):
            break
        try:
            out_sr, samples = speak(model, nxt, sf, torch)
            reply(out_sr, samples)
        except Exception as exc:                       # noqa: BLE001 - reported, not raised
            import traceback
            print(json.dumps({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc),
                              "traceback": traceback.format_exc()[-1500:]}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:                       # one JSON line either way
        import traceback
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                          "traceback": traceback.format_exc()[-3000:]}), flush=True)
        sys.exit(1)
