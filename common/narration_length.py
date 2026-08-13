"""Set a shot's length from the narration that will play over it.

The scene plan can only *estimate* how long a line takes to say — words per second varies
with the sentence, the voice and the language. When the estimate runs short the clip ends
before the sentence does, and `ArkVideoDub` has to freeze the last frame to cover the
remainder. Measured: an 11.10 s narration over a 5.17 s shot meant more than half the
scene was a still image.

Measuring the rendered narration removes the guess. This node runs after `ArkQwenTTS` and
before `ArkHailuoScene`, so by the time the shot is sampled its length is known exactly,
and the picture always outlasts the voice.

Wiring it also *enforces* that order: `length` becomes a dependency of the audio, so
ComfyUI cannot schedule the expensive H3 pass before the cheap TTS pass.

MiniMax H3 constrains the answer. Frames must land on a 17k+5 grid, and the trained range
is roughly 124-362 frames — about 5.2 s to 15.1 s at 24 fps. A narration longer than that
cannot be covered by any single shot, so the node clamps and says so; the fix for that
case belongs in the brief, which should cap how much is written per scene.
"""

# LOCKED to ArkHailuoScene.FPS and ArkSceneList.FPS. H3 works in frames at 24 fps.
FPS = 24.0

# H3's trained window. Below MIN the model degrades rather than just costing less; above
# MAX it drifts. Both sit on the 17k+5 grid: 124 = 17*7+5, 362 = 17*21+5.
MIN_FRAMES, MAX_FRAMES = 124, 362


def snap_length(seconds, fps=FPS):
    """Seconds -> a frame count H3 accepts, rounded UP so speech is never clipped."""
    length = max(5, int(-(-seconds * fps // 1)))       # ceil
    length += (5 - (length % 17)) % 17                 # onto the 17k+5 grid
    return max(MIN_FRAMES, min(MAX_FRAMES, length))


class ArkNarrationLength:
    CATEGORY = "arkennemasis/Video"
    FUNCTION = "run"
    # `seconds_int` is APPENDED, never inserted: output slots are positional, and adding
    # one in the middle would silently repoint every existing link on this node.
    # It exists because video models disagree about how a shot is measured — MiniMax H3
    # wants FRAMES on its 17k+5 grid, LTX-2.5 wants whole SECONDS, and a FLOAT will not
    # connect to an INT socket.
    RETURN_TYPES = ("INT", "FLOAT", "STRING", "INT")
    RETURN_NAMES = ("length", "seconds", "report", "seconds_int")
    DESCRIPTION = ("Measure a narration clip and return the shot length that covers it, "
                   "snapped to MiniMax H3's frame grid. Wire between the TTS node and "
                   "the scene node so every shot outlasts its own voice-over.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "narration": ("AUDIO", {
                    "tooltip": "The rendered voice-over for THIS scene.",
                }),
                "tail_seconds": ("FLOAT", {
                    "default": 0.75, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Picture held after the last word. Cutting on the "
                               "syllable reads as a mistake; a beat reads as editing.",
                }),
            },
            "optional": {
                "max_seconds": ("FLOAT", {
                    "default": 15.0, "min": 5.2, "max": 15.1, "step": 0.1,
                    "tooltip": "Hard ceiling. MiniMax H3 cannot exceed about 15.1 s in "
                               "one shot, so a longer narration has to be shortened in "
                               "the brief or split across two scenes.",
                }),
                "lock_seconds": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 15.1, "step": 0.5,
                    "tooltip": "0 = fit each shot to its own narration. Above 0, EVERY "
                               "shot runs this long regardless of what is said — even "
                               "pacing, at the cost of holding a frame under a short "
                               "line. The value still snaps to H3's frame grid, so 10 s "
                               "becomes 10.12 s. Keep the brief's word count under about "
                               "2 words per second of the locked length, or the voice "
                               "will outrun the picture.",
                }),
            },
        }

    def run(self, narration, tail_seconds=0.75, max_seconds=15.0, lock_seconds=0.0):
        waveform = narration["waveform"]
        rate = float(narration["sample_rate"]) or 1.0
        spoken = waveform.shape[-1] / rate

        if lock_seconds and lock_seconds > 0:
            # Every shot the same length. The narration is still measured, because a line
            # that overruns a locked shot is the one thing the operator needs told — the
            # dub will hold a frame over the difference and it will look like a fault.
            length = snap_length(float(lock_seconds))
            actual = length / FPS
            report = ("locked %.2fs -> %d frames (%.2fs) | narration %.2fs"
                      % (lock_seconds, length, actual, spoken))
            if spoken > actual:
                report += ("  WARNING: %.2fs of speech in a %.2fs shot — the last frame "
                           "will freeze for %.2fs. Shorten this line to about %d words."
                           % (spoken, actual, spoken - actual, int(actual * 2.0)))
            print("[arkennemasis] %s" % report)
            return (length, actual, report, int(round(actual)))

        wanted = spoken + float(tail_seconds)
        length = snap_length(min(wanted, float(max_seconds)))
        actual = length / FPS

        report = ("narration %.2fs + %.2fs tail -> %d frames (%.2fs)"
                  % (spoken, tail_seconds, length, actual))
        if actual < spoken:
            # Say it plainly rather than letting the dub silently freeze-frame over it.
            report += ("  WARNING: %.2fs of speech cannot fit in one H3 shot (max %.2fs) "
                       "— shorten this scene's voiceText or split the scene"
                       % (spoken, MAX_FRAMES / FPS))
        print("[arkennemasis] %s" % report)
        return (length, actual, report, int(round(actual)))


NODE_CLASS_MAPPINGS = {
    "ArkNarrationLength": ArkNarrationLength,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkNarrationLength": "arkennemasis Narration Length (fit the shot to the voice)",
}
