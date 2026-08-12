"""Run Qwen3-TTS in a subprocess that sees transformers 4.57.3 instead of 5.x.

Qwen3-TTS is written against transformers 4.57.3. This ComfyUI install runs 5.14.1, and
eleven other node packs depend on that, so downgrading in place is not an option. Shimming
5.x back to the 4.x surface got the model loading but then diverged inside attention —
porting transformers internals is not a maintainable answer.

So the model runs in its own process with a private `transformers` first on `sys.path`.
Same interpreter, same torch, same CUDA — only `transformers` differs, and only here.
Nothing about the main environment changes, which is why this survives being shared.

Invoked as::

    python qwen_tts_worker.py <job.json>

`job.json` carries model_dir, text, language, an optional reference wav path and its
transcript, and where to write the result. The reply is one JSON line on stdout.
"""

from __future__ import annotations

import json
import os
import sys


def main() -> int:
    job = json.loads(open(sys.argv[1], "r", encoding="utf-8").read())

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

    torch.manual_seed(int(job.get("seed") or 0))
    language = job.get("language") or None
    text = job["text"]

    ref_path = job.get("ref_audio")
    if ref_path:
        import librosa
        wav, sr = librosa.load(ref_path, sr=None, mono=True)
        ref_text = (job.get("ref_text") or "").strip() or None
        wavs, out_sr = model.generate_voice_clone(
            text=text, language=language, ref_audio=(wav, sr), ref_text=ref_text,
            x_vector_only_mode=ref_text is None)
    else:
        speaker = job.get("speaker") or None
        wavs, out_sr = model.generate_custom_voice(
            text=text, language=language, speaker=speaker)

    audio = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
    sf.write(job["out_path"], audio, out_sr)
    print(json.dumps({"ok": True, "sample_rate": int(out_sr),
                      "samples": int(len(audio)),
                      "device": device,
                      "gpu": (torch.cuda.get_device_name(0)
                              if device == "cuda" else None),
                      "transformers": transformers.__version__}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:                       # one JSON line either way
        import traceback
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                          "traceback": traceback.format_exc()[-3000:]}), flush=True)
        sys.exit(1)
