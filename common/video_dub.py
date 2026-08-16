"""Replace a clip's soundtrack with a narration track.

A narrated film wants a chosen voice over the shot, but MiniMax H3 always generates its
own audio — it is an omni-modal model and cannot be asked for silence. So the clip arrives
with the wrong soundtrack and the narration arrives separately, and something has to put
them together before the clips are concatenated.

Doing it per clip rather than over the finished film matters: each scene's narration has
to line up with its own shot, and once the clips are joined the boundaries are gone.

The video is stretched or trimmed to the narration, not the other way round — speech that
gets cut off mid-word reads as broken, whereas a shot that holds a moment longer does not.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

from .video_assemble import find_ffmpeg


def _video_path(video, given: str = "") -> str:
    """A VIDEO object -> a file on disk.

    `given` wins: a node that already wrote the clip knows its path, and asking is more
    reliable than introspecting. `ArkHailuoScene` returns exactly that on its second
    output.

    Falling back to `save_to()` is a last resort — it re-encodes through TorchCodec,
    which is not installed here, so a run that reached it failed after every shot had
    already rendered.
    """
    if given and os.path.exists(given):
        return given

    for attr in ("_VideoFromFile__file", "file", "path", "_path"):
        value = getattr(video, attr, None)
        if isinstance(value, str) and os.path.exists(value):
            return value
    source = getattr(video, "get_stream_source", None)
    if callable(source):
        try:
            value = source()
            if isinstance(value, str) and os.path.exists(value):
                return value
        except Exception:
            pass

    handle, path = tempfile.mkstemp(suffix=".mp4", prefix="ark_dub_in_")
    os.close(handle)
    try:
        video.save_to(path)
    except Exception as exc:
        raise RuntimeError(
            "ArkVideoDub could not get a file for this clip: "
            f"{type(exc).__name__}: {exc}. Wire the producing node's `path` output into "
            "`video_path` — ArkHailuoScene provides one.") from exc
    return path


class ArkVideoDub:
    CATEGORY = "arkennemasis/Video"
    FUNCTION = "run"
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "report")
    DESCRIPTION = (
        "Swap a clip's audio for a narration track. The clip is held to the length of "
        "the narration so no word is cut off."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "narration": ("AUDIO",),
                "fit": (["hold last frame", "cut video to narration", "keep video length"], {
                    "tooltip": "What to do when the narration and the shot differ in "
                               "length. 'hold last frame' never truncates speech.",
                }),
            },
            "optional": {
                "video_path": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "The clip's file on disk. Wire the producing node's "
                               "`path` output here — reading the file directly avoids a "
                               "re-encode through TorchCodec, which is not installed.",
                }),
                "keep_original_at": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Mix the clip's own audio back in underneath, 0 = drop it "
                               "entirely. Useful for keeping a little room tone.",
                }),
            },
        }

    def run(self, video, narration, fit, video_path="", keep_original_at=0.0):
        # soundfile, NOT torchaudio.save: torchaudio 2.11 routes `save` through
        # `save_with_torchcodec`, and TorchCodec is not installed here. That import error
        # surfaces as "TorchCodec is required" and reads like a video problem, which is
        # what it was mistaken for — it is the narration being written out.
        import soundfile as sf

        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError("ArkVideoDub: no ffmpeg found on PATH or in imageio-ffmpeg.")

        in_path = _video_path(video, video_path)

        handle, wav_path = tempfile.mkstemp(suffix=".wav", prefix="ark_dub_voice_")
        os.close(handle)
        waveform = narration["waveform"]
        if waveform.dim() == 3:                 # (batch, channels, samples)
            waveform = waveform[0]
        # torch is (channels, samples); soundfile wants (frames, channels).
        samples = waveform.detach().cpu().float().numpy().T
        sf.write(wav_path, samples, int(narration["sample_rate"]))

        handle, out_path = tempfile.mkstemp(suffix=".mp4", prefix="ark_dub_out_")
        os.close(handle)

        cmd = [ffmpeg, "-y", "-i", in_path, "-i", wav_path]
        if keep_original_at > 0:
            # Duck the original under the voice rather than replacing it.
            cmd += ["-filter_complex",
                    f"[0:a]volume={keep_original_at}[bed];[bed][1:a]amix=inputs=2:"
                    f"duration=longest:dropout_transition=0[a]",
                    "-map", "0:v", "-map", "[a]"]
        else:
            cmd += ["-map", "0:v", "-map", "1:a"]

        if fit == "hold last frame":
            # tpad freezes the final frame so the picture never runs out before the voice.
            cmd += ["-vf", "tpad=stop_mode=clone:stop_duration=600", "-shortest"]
        elif fit == "cut video to narration":
            cmd += ["-shortest"]
        else:
            # "keep video length": the shot is a fixed slot and the line is shorter, so the
            # rest of the slot must be SILENCE — not a short audio stream.
            #
            # That distinction is the whole bug. Left alone, the clip carries 10 s of video
            # and 6 s of audio, which plays fine on its own and then falls apart when the
            # clips are joined: ffmpeg's concat demuxer expects every clip's streams to run
            # the same length, so an audio stream that ends early shifts the timestamps of
            # everything after it and narration starts landing under the wrong scene.
            #
            # `apad` extends the audio with real silence and `-shortest` cuts it at the end
            # of the picture, so audio and video are exactly equal and concat has nothing
            # to drift. The gap is silent, which is what was asked for.
            cmd += (["-af", "apad", "-shortest"] if keep_original_at <= 0
                    else ["-filter_complex", cmd[cmd.index("-filter_complex") + 1] + ",apad",
                          "-shortest"])

        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", out_path]

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0 or not os.path.exists(out_path):
            raise RuntimeError("ArkVideoDub: ffmpeg failed:\n"
                               + proc.stderr.decode("utf-8", "replace")[-1500:])

        seconds = waveform.shape[-1] / float(narration["sample_rate"] or 1)
        report = f"dubbed {seconds:.2f}s of narration onto the clip ({fit})"
        print(f"[arkennemasis] {report}")

        from comfy_api.latest._input_impl.video_types import VideoFromFile
        return (VideoFromFile(out_path), report)


NODE_CLASS_MAPPINGS = {
    "ArkVideoDub": ArkVideoDub,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkVideoDub": "arkennemasis Video Dub (narration over a clip)",
}
