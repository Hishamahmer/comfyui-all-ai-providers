"""Land every narration on the same mark.

A locked shot gives each line the same slot, and the brief gives each line the same
word budget — and neither controls the thing that varies most. Measured over one real
run of this voice, the SAME kind of 15-17 word line was delivered anywhere between
1.88 and 2.83 words a second: a 50% spread, and a 3.4 s spread on an 8 s target. So a
line that is the right length still finishes wherever it finishes, and the quiet beat
before the next room is 1.5 s on one shot and 4.4 s on the next.

Asking the writer for shorter lines only moves the average. This moves the actual
landing: the rendered speech is stretched or compressed to end on `target_seconds`,
pitch preserved, so every shot speaks for the same time and every pause is the same
length.

It is deliberately NOT unlimited. Past about 15% the change stops being inaudible and
starts sounding sluggish or hurried, which is worse than an uneven pause — so the
factor is clamped, and when it clamps the node SAYS so rather than quietly delivering
something outside the band it promised.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import torch

from .video_assemble import find_ffmpeg


def _atempo_chain(factor):
    """ffmpeg's atempo accepts 0.5-2.0 per instance; chain them for anything wider.

    The clamp below keeps us far inside one instance, but the chain costs nothing and
    means a future caller raising `max_change` cannot silently get a filter error.
    """
    parts = []
    remaining = float(factor)
    while remaining > 2.0:
        parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    parts.append("atempo=%.6f" % remaining)
    return ",".join(parts)


class ArkNarrationFit:
    CATEGORY = "arkennemasis/Video"
    FUNCTION = "run"
    RETURN_TYPES = ("AUDIO", "FLOAT", "STRING")
    RETURN_NAMES = ("audio", "seconds", "report")
    DESCRIPTION = ("Stretch or compress a rendered narration so it ends exactly on a "
                   "target, pitch preserved. Use it when every shot is the same length "
                   "and the pause after the words has to be the same too.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "narration": ("AUDIO", {
                    "tooltip": "The rendered voice-over for THIS scene.",
                }),
                "target_seconds": ("FLOAT", {
                    "default": 8.0, "min": 0.5, "max": 60.0, "step": 0.25,
                    "tooltip": "Where the last word should land. With a locked shot, "
                               "this is (shot length - the quiet beat you want after it).",
                }),
                "max_change": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "How far the speed may move, as a fraction. 0.15 = at "
                               "most 15% faster or slower, which is inaudible on speech. "
                               "Beyond about 0.2 it starts to sound wrong; the node "
                               "clamps and reports rather than mangling the read.",
                }),
                "tolerance": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Leave the audio untouched when it already lands this "
                               "close. Avoids a needless re-encode on a line that was "
                               "already the right length.",
                }),
            },
        }

    def run(self, narration, target_seconds=8.0, max_change=0.15, tolerance=0.15):
        import soundfile as sf

        waveform = narration["waveform"]
        rate = int(narration["sample_rate"]) or 24000
        work = waveform[0] if waveform.dim() == 3 else waveform
        spoken = work.shape[-1] / float(rate)

        if spoken <= 0:
            report = "ArkNarrationFit: empty narration, nothing to fit"
            print("[arkennemasis] %s" % report)
            return (narration, 0.0, report)

        target = float(target_seconds)
        if abs(spoken - target) <= float(tolerance):
            report = ("already lands at %.2fs (target %.2fs, within %.2fs) — untouched"
                      % (spoken, target, tolerance))
            print("[arkennemasis] narration fit: %s" % report)
            return (narration, spoken, report)

        # atempo > 1 speeds up. To make a 6.0s line last 8.0s we need to SLOW it, so the
        # factor is spoken/target, not the other way round — getting this inverted makes
        # every short line shorter still, which looks like the node doing nothing.
        wanted = spoken / target
        low, high = 1.0 - float(max_change), 1.0 + float(max_change)
        factor = max(low, min(high, wanted))
        clamped = abs(factor - wanted) > 1e-6

        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError("ArkNarrationFit: no ffmpeg found on PATH or in "
                               "imageio-ffmpeg.")

        handle, src = tempfile.mkstemp(suffix=".wav", prefix="ark_fit_in_")
        os.close(handle)
        handle, dst = tempfile.mkstemp(suffix=".wav", prefix="ark_fit_out_")
        os.close(handle)
        try:
            # soundfile, not torchaudio.save: torchaudio 2.11 routes save through
            # TorchCodec, which is not installed here. Same reason as ArkVideoDub.
            sf.write(src, work.detach().cpu().float().numpy().T, rate)
            cmd = [ffmpeg, "-y", "-v", "error", "-i", src,
                   "-filter:a", _atempo_chain(factor), dst]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
            data, out_rate = sf.read(dst, dtype="float32", always_2d=True)
        finally:
            for path in (src, dst):
                try:
                    os.remove(path)
                except OSError:
                    pass

        fitted = torch.from_numpy(data.T).unsqueeze(0)
        landed = fitted.shape[-1] / float(out_rate)
        report = ("%.2fs -> %.2fs (target %.2fs, speed x%.3f)"
                  % (spoken, landed, target, factor))
        if clamped:
            report += ("  NOTE: wanted x%.3f but capped at %d%%. The line is too %s for "
                       "this slot — change the word budget, not this node."
                       % (wanted, round(max_change * 100),
                          "long" if wanted > high else "short"))
        print("[arkennemasis] narration fit: %s" % report)
        return ({"waveform": fitted, "sample_rate": int(out_rate)}, landed, report)


NODE_CLASS_MAPPINGS = {
    "ArkNarrationFit": ArkNarrationFit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkNarrationFit": "arkennemasis Narration Fit (land every line on the same mark)",
}
