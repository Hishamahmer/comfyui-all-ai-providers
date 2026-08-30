"""How many frames the presenter clip must be, measured from the voice that will play over it.

The talking-head model is driven by audio: the voice decides how long the clip is, and the
clip has to cover every word or the sentence is cut off mid-syllable. So this sits between
the TTS node and the sampler and turns the rendered narration into a frame count.

Wiring it also **pins the execution order**. The frame count becomes a dependency of the
audio, so ComfyUI cannot schedule the expensive video pass before the cheap voice pass —
which means a script that fails to speak fails in seconds instead of after a render.

**Why not `ArkNarrationLength`.** That node does the same job for MiniMax H3 and is wrong
here in three ways that cannot be reconciled by a widget. H3 wants frames on a 17k+5 grid;
Wan's VAE compresses time by 4, so it wants **4k+1**. H3's trained window clamps to
124-362 frames, about 5 to 15 seconds — a presenter reading a 30-second script would be
truncated to a third of it. And H3 never needed a cost ceiling, because it could not
exceed 15 seconds anyway. Widening `ArkNarrationLength` to cover both would change the
numbers under the walkthrough canvas that already depends on it.

**`max_seconds` is a cost control, not a safety rail.** Sampling cost grows faster than
linearly with clip length — attention is quadratic in the sequence — so doubling the
script does considerably more than double the render. The cap stops a model that wrote
three paragraphs from quietly committing the machine to hours, and it says so in the
report rather than failing silently.
"""

from __future__ import annotations

# Wan's VAE compresses time 4:1, so a valid frame count is 4k+1. Handing it anything else
# means the sampler silently works to a different length than the audio embeddings were
# built for, and the lip-sync drifts further with every second.
GRID = 4

# The Zohran MultiTalk graph runs at 25 fps end to end — the wav2vec embeds, the sampler
# and the muxer all agree on it. Changing it here alone would desynchronise them, so it is
# a widget with a default rather than a constant, and the builder wires the same number
# into every consumer.
DEFAULT_FPS = 25.0


def snap(seconds, fps=DEFAULT_FPS):
    """Seconds -> a frame count Wan accepts, rounded UP so speech is never clipped."""
    frames = max(1, int(-(-float(seconds) * float(fps) // 1)))      # ceil
    # Onto the 4k+1 grid, upward for the same reason.
    remainder = (frames - 1) % GRID
    if remainder:
        frames += GRID - remainder
    return frames


class ArkAvatarFrames:
    CATEGORY = "arkennemasis/Avatar"
    FUNCTION = "run"
    RETURN_TYPES = ("INT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("num_frames", "seconds", "fps", "report")
    DESCRIPTION = ("Measure the narration and return the frame count that covers it, on "
                   "Wan's 4k+1 grid. Wire between the TTS node and the video model so "
                   "the clip is always at least as long as the voice.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "narration": ("AUDIO", {
                    "tooltip": "The rendered voice-over. Its real length is what sizes "
                               "the clip — not the word count it was estimated from.",
                }),
                "fps": ("FLOAT", {
                    "default": DEFAULT_FPS, "min": 8.0, "max": 60.0, "step": 1.0,
                    "tooltip": "Frames per second. Must match the audio-embedding node "
                               "and the muxer, or the lips drift from the voice.",
                }),
                "tail_seconds": ("FLOAT", {
                    "default": 0.4, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Picture held after the last word. Cutting on the "
                               "syllable reads as a mistake; a beat reads as editing.",
                }),
                "max_seconds": ("FLOAT", {
                    "default": 45.0, "min": 2.0, "max": 300.0, "step": 1.0,
                    "tooltip": "Hard ceiling on the clip. Render time grows faster than "
                               "linearly with length, so this is the difference between "
                               "a long render and an overnight one. Exceeding it "
                               "truncates the picture, and the report says so.",
                }),
            },
        }

    def run(self, narration, fps=DEFAULT_FPS, tail_seconds=0.4, max_seconds=45.0):
        waveform = narration.get("waveform") if isinstance(narration, dict) else None
        if waveform is None:
            raise RuntimeError("ArkAvatarFrames needs an AUDIO input with a waveform.")
        rate = float(narration.get("sample_rate") or 0) or 1.0
        spoken = float(waveform.shape[-1]) / rate
        if spoken <= 0:
            raise RuntimeError(
                "ArkAvatarFrames: the narration is zero-length. The TTS produced no "
                "audio — check its report before spending a render on this.")

        wanted = spoken + max(0.0, float(tail_seconds))
        capped = min(wanted, float(max_seconds))
        frames = snap(capped, fps)
        seconds = frames / float(fps)

        note = ""
        if wanted > float(max_seconds) + 1e-6:
            note = ("  | CAPPED: the voice is %.1fs and the ceiling is %.1fs, so the "
                    "last %.1fs will have no picture. Shorten the script or raise "
                    "max_seconds." % (spoken, max_seconds, wanted - max_seconds))
            print("[arkennemasis] avatar frames:%s" % note)

        report = ("voice %.2fs (+%.2fs tail) -> %d frames = %.2fs at %.0f fps%s"
                  % (spoken, tail_seconds, frames, seconds, fps, note))
        print("[arkennemasis] avatar frames: %s" % report)
        return (frames, seconds, float(fps), report)


NODE_CLASS_MAPPINGS = {"ArkAvatarFrames": ArkAvatarFrames}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkAvatarFrames": "arkennemasis Avatar Frames (clip length from the voice)",
}
