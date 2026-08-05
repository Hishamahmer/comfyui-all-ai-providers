"""Join scene clips into one video: concat + music bed + burned-in subtitles.

The aggregate end of the loop. ``ArkSceneList`` fans the plan out so the scene chain
runs once per scene; this node declares ``INPUT_IS_LIST`` so ComfyUI hands it the WHOLE
collection of finished clips in a single call, in iteration order, however many there
are. That is the n8n Aggregate step, and it is what lets one canvas serve 5 or 50 scenes.

Because INPUT_IS_LIST makes *every* input arrive as a list, the scalar settings come in
wrapped too and are unwrapped with `_one`.

Doing the join locally also fixes two things a hosted ffmpeg service cannot: the music
bed is trimmed to the video length and ducked under the dialogue, and each clip's speech
is levelled to broadcast loudness. Video models mix speech very quietly — measured -20
to -27 dB on MiniMax H3 output against about -16 LUFS for normal web video — so without
levelling the finished cut sounds nearly mute.

Subtitles are timed from the SAME plan that generated the video and each clip's real
duration, so cue N starts exactly where clips 1..N-1 ended. Strictly better than
transcribing the finished audio and hoping the words line up.
"""

import json
import os
import shutil
import subprocess
import tempfile


def _one(value, default=None):
    """First element of an INPUT_IS_LIST-wrapped scalar."""
    if isinstance(value, list):
        return value[0] if value else default
    return default if value is None else value


def find_ffmpeg():
    """ffmpeg from PATH, else the copy imageio-ffmpeg ships inside the embedded python."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


# The subtitles filter parses its own argument string, so a Windows drive colon reads as
# an option separator: ffmpeg took `C:/…/subs.srt` as filename `C` plus an option
# `original_size=/…/subs.srt` and refused it. Escaping the colon is fragile across ffmpeg
# versions and shells, so instead we run ffmpeg WITH ITS CWD SET to the work directory
# and name the file bare — no colon, no separators, nothing to escape.
SUBS_NAME = "subs.srt"


def _srt_time(seconds):
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return "%02d:%02d:%02d,%03d" % (h, m, s, ms)


def _wrap(text, width=42):
    words, lines, line = str(text).split(), [], ""
    for word in words:
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = (line + " " + word).strip()
    if line:
        lines.append(line)
    return "\n".join(lines[:2])          # two lines max; more covers the frame


def _run(cmd, cwd=None):
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        # ffmpeg prints its whole build config before the error, so show the TAIL —
        # the first 2000 chars are always the same --enable-lib... wall.
        tail = proc.stdout.decode("utf-8", "replace")[-2000:]
        raise RuntimeError("ffmpeg failed (%d):\n%s" % (proc.returncode, tail))


class ArkVideoAssemble:
    CATEGORY = "arkennemasis/Video"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "VIDEO")
    RETURN_NAMES = ("final_path", "video")
    OUTPUT_NODE = True
    INPUT_IS_LIST = True          # receive every clip of the run in one call
    DESCRIPTION = ("Join every scene clip into one video: concatenate in order, level "
                   "the speech, trim and duck a music bed, and burn subtitles taken "
                   "from the scene plan. Collects the whole loop in one call.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "videos": ("VIDEO", {
                    "tooltip": "Wire the scene chain's video output here. The loop "
                               "delivers every scene to this one socket, in order.",
                }),
                "output_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Where to write final.mp4. Relative paths resolve inside "
                               "ComfyUI's output directory. Wire from Run Folder.",
                }),
                "filename": ("STRING", {"default": "final"}),
            },
            "optional": {
                "scenes_json": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "The same plan that made the clips. Its voiceText fields "
                               "become the subtitles, timed from each clip's real "
                               "duration.",
                }),
                "music_path": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute path to a music file. Blank = no bed. Trimmed "
                               "to the video length automatically.",
                }),
                "music_volume": ("FLOAT", {
                    "default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "0.18 sits under spoken dialogue. The clips carry their "
                               "own speech, so this must stay low.",
                }),
                "normalize_speech": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Level each clip to EBU R128 -16 LUFS. Without it the "
                               "finished cut sounds nearly mute.",
                }),
                "burn_subtitles": ("BOOLEAN", {"default": True}),
                "subtitle_size": ("INT", {"default": 24, "min": 8, "max": 200}),
                "crf": ("INT", {
                    "default": 18, "min": 0, "max": 51,
                    "tooltip": "x264 quality; lower is better. 18 is visually lossless.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")          # always re-assemble; the clips may have changed

    def run(self, videos, output_dir, filename, scenes_json=None, music_path=None,
            music_volume=None, normalize_speech=None, burn_subtitles=None,
            subtitle_size=None, crf=None):
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError(
                "ArkVideoAssemble: no ffmpeg found. Put it on PATH, or install "
                "imageio-ffmpeg into the embedded python.")

        clips = [v for v in (videos or []) if v is not None]
        if not clips:
            raise ValueError("ArkVideoAssemble: no clips arrived. Is the scene chain "
                             "wired into `videos`?")

        out_dir = _one(output_dir, "")
        name = os.path.basename(str(_one(filename, "final")).strip() or "final")
        plan = _one(scenes_json, "") or ""
        music = str(_one(music_path, "") or "").strip()
        volume = _one(music_volume, 0.18)
        level = _one(normalize_speech, True)
        subs_on = _one(burn_subtitles, True)
        subs_size = int(_one(subtitle_size, 24))
        quality = int(_one(crf, 18))

        folder = str(out_dir).strip()
        if not folder or not os.path.isabs(folder):
            try:
                import folder_paths
                base = folder_paths.get_output_directory()
            except Exception:
                base = os.getcwd()
            folder = os.path.join(base, folder) if folder else base
        os.makedirs(folder, exist_ok=True)

        work = tempfile.mkdtemp(prefix="ark_assemble_")
        try:
            paths, durations = [], []
            for index, video in enumerate(clips, start=1):
                clip = os.path.join(work, "clip_%03d.mp4" % index)
                # save_to re-encodes from whatever the VIDEO actually is, so this works
                # for a clip built in-graph (VideoFromComponents, no file behind it) as
                # well as one loaded from disk.
                video.save_to(clip)
                if level:
                    # Level EACH clip, not the joined file: one pass over the whole
                    # concatenation would leave a quiet scene quiet relative to a loud
                    # one. Video is stream-copied, so this cannot change a pixel.
                    leveled = os.path.join(work, "clip_%03d_n.mp4" % index)
                    try:
                        _run([ffmpeg, "-y", "-i", clip, "-c:v", "copy",
                              "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                              "-c:a", "aac", "-b:a", "192k", leveled])
                        clip = leveled
                    except RuntimeError as exc:
                        print("[arkennemasis] loudnorm skipped for clip %d (%s)"
                              % (index, str(exc)[:120]))
                paths.append(clip)
                try:
                    durations.append(float(video.get_duration()))
                except Exception:
                    durations.append(0.0)

            print("[arkennemasis] assembling %d clips (%.1fs total)%s"
                  % (len(paths), sum(durations),
                     ", speech normalised to -16 LUFS" if level else ""))

            listing = os.path.join(work, "clips.txt")
            with open(listing, "w", encoding="utf-8", newline="\n") as f:
                for path in paths:
                    f.write("file '%s'\n" % path.replace("\\", "/"))

            joined = os.path.join(work, "joined.mp4")
            _run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", listing,
                  "-c:v", "libx264", "-crf", str(quality), "-preset", "medium",
                  "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", joined])

            srt = self._write_srt(work, plan, durations) if (subs_on and plan) else None

            final = os.path.join(folder, "%s.mp4" % name)
            cmd = [ffmpeg, "-y", "-i", joined]
            filters, maps = [], []
            if music and os.path.exists(music):
                cmd += ["-i", music]
                filters.append(
                    "[1:a]volume=%s[bed];"
                    "[0:a][bed]amix=inputs=2:duration=first:dropout_transition=0[aout]"
                    % volume)
                maps = ["-map", "0:v" if not srt else "[vout]", "-map", "[aout]"]
            elif music:
                print("[arkennemasis] music file not found, skipping bed: %s" % music)

            if srt:
                # Bare filename only — see SUBS_NAME above. ffmpeg runs with cwd=work.
                filters.append(
                    "[0:v]subtitles=%s:force_style='FontSize=%d,PrimaryColour=&Hffffff,"
                    "OutlineColour=&H80000000,BorderStyle=3,Outline=1,Shadow=0,"
                    "Alignment=2,MarginV=28'[vout]" % (SUBS_NAME, subs_size))
                if not maps:
                    maps = ["-map", "[vout]", "-map", "0:a?"]

            if filters:
                cmd += ["-filter_complex", ";".join(filters)] + maps
            cmd += ["-c:v", "libx264", "-crf", str(quality), "-preset", "medium",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k"]
            if music and os.path.exists(music):
                cmd.append("-shortest")
            cmd.append(final)
            _run(cmd, cwd=work)

            print("[arkennemasis] final video -> %s (%.1f MB)"
                  % (final, os.path.getsize(final) / 1e6))
        finally:
            shutil.rmtree(work, ignore_errors=True)

        from comfy_api.latest._input_impl.video_types import VideoFromFile
        result = (final, VideoFromFile(final))

        # Report the file so the finished video appears in the Job Queue and can be
        # played there; without this the node runs silently.
        try:
            import folder_paths
            rel = os.path.relpath(final, folder_paths.get_output_directory())
            if not rel.startswith(".."):
                return {"ui": {"images": [{"filename": os.path.basename(final),
                                           "subfolder": os.path.dirname(rel).replace(
                                               "\\", "/"),
                                           "type": "output"}]},
                        "result": result}
        except Exception:
            pass
        return result

    def _write_srt(self, work, scenes_json, durations):
        try:
            scenes = json.loads(scenes_json)
        except ValueError:
            print("[arkennemasis] scenes_json unparseable — skipping subtitles")
            return None
        if isinstance(scenes, dict):
            scenes = scenes.get("scenes") or scenes.get("output") or []
        if not isinstance(scenes, list) or not scenes:
            return None

        lines, clock, cue = [], 0.0, 1
        # Positional pairing: clip i is scene i, because the loop preserves order.
        for scene, duration in zip(scenes, durations):
            text = scene.get("voiceText", "") if isinstance(scene, dict) else ""
            if text and duration > 0:
                lines.append("%d\n%s --> %s\n%s\n"
                             % (cue, _srt_time(clock), _srt_time(clock + duration),
                                _wrap(text)))
                cue += 1
            clock += duration
        if not lines:
            return None
        path = os.path.join(work, SUBS_NAME)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines))
        print("[arkennemasis] %d subtitle cues from the scene plan" % (cue - 1))
        return path


NODE_CLASS_MAPPINGS = {
    "ArkVideoAssemble": ArkVideoAssemble,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkVideoAssemble": "arkennemasis Video Assemble (clips + music + subs)",
}
