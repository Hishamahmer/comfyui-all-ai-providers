"""Join scene clips into one video: concat + music bed + burned-in subtitles.

The aggregate end of the loop. ``ArkSceneList`` fans the plan out so the scene chain
runs once per scene; this node declares ``INPUT_IS_LIST`` so ComfyUI hands it the WHOLE
collection of finished clips in a single call, in iteration order, however many there
are. That is what lets one canvas serve 5 scenes or 50.

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

Wire a ``Caption Style`` node into `caption_style` to choose the font and one of the
five subtitle styles; the cues then go out as ASS instead of SRT. Leave it unconnected
and you get the plain white captions this node has always burned.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile

from . import ass_captions as ass


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


def find_ffprobe():
    exe = shutil.which("ffprobe")
    if exe:
        return exe
    ffmpeg = find_ffmpeg()          # ffprobe normally sits in the same bin directory
    if ffmpeg:
        guess = os.path.join(os.path.dirname(ffmpeg),
                             "ffprobe.exe" if os.name == "nt" else "ffprobe")
        if os.path.exists(guess):
            return guess
    return None


def probe_speech(path, fallback):
    """Where the NARRATION stops inside a clip, which is not where the clip stops.

    A shot is a fixed slot and the line is shorter, so the dub pads the rest with silence:
    audio and video are the same length (concat needs that), and the tail is quiet. So the
    audio stream's duration no longer says anything about the speech, and captions spread
    across the slot would drift seconds behind the voice.

    ffmpeg's `silencedetect` finds the quiet tail. If the last silent stretch runs to the
    end of the clip, the voice stopped where that silence began. Anything else — a clip
    trimmed to its narration, a probe that fails, a detector that finds nothing — falls
    back to the clip length, which is correct for those cases.
    """
    ffmpeg = find_ffmpeg()
    if not ffmpeg or fallback <= 0:
        return fallback
    try:
        out = subprocess.run(
            [ffmpeg, "-hide_banner", "-nostats", "-i", path,
             "-af", "silencedetect=noise=-45dB:d=0.35", "-f", "null", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
        log = out.stderr.decode("utf-8", "replace")
    except Exception:
        return fallback

    starts = [float(m) for m in re.findall(r"silence_start:\s*(-?[\d.]+)", log)]
    ends = [float(m) for m in re.findall(r"silence_end:\s*([\d.]+)", log)]
    if not starts:
        return fallback
    last = starts[-1]
    # Only a silence that runs to the END of the clip marks the end of speech. A pause in
    # the middle has a matching silence_end after it and must not truncate the captions.
    closed = [e for e in ends if e > last]
    if closed and closed[-1] < fallback - 0.2:
        return fallback
    speech = max(0.0, min(last, fallback))
    return speech if speech > 0.3 else fallback


def probe_size(path, default=(1280, 720)):
    """(width, height) of a video, for PlayResX/Y and the auto font size."""
    probe = find_ffprobe()
    if probe:
        try:
            out = subprocess.run(
                [probe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=width,height", "-of", "csv=p=0:s=x", path],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL, timeout=30)   # never inherit the console
            w, h = out.stdout.decode().strip().split("x")[:2]
            return int(w), int(h)
        except Exception:
            pass
    print("[arkennemasis] could not probe %s — assuming %dx%d for subtitle sizing"
          % (os.path.basename(path), default[0], default[1]))
    return default


# The subtitles filter parses its own argument string, so a Windows drive colon reads as
# an option separator: ffmpeg took `C:/…/subs.srt` as filename `C` plus an option
# `original_size=/…/subs.srt` and refused it. Escaping the colon is fragile across ffmpeg
# versions and shells, so instead we run ffmpeg WITH ITS CWD SET to the work directory
# and name the file bare — no colon, no separators, nothing to escape.
SUBS_NAME = "subs.srt"
SUBS_ASS = "subs.ass"

# Speech rarely fills a generated clip end to end, so the estimated word timings are
# spread across an inset window rather than the whole shot. Measured against MiniMax H3
# output: roughly a quarter-second of lead-in and a longer tail as the shot holds.
LEAD_IN, TAIL = 0.25, 0.35


def _voice_text(scene):
    """This shot's narration, under either spelling.

    Matches ``ArkSceneList``: a brief written consistently in snake_case still lands,
    instead of producing silent clips with no subtitles and no error.
    """
    if not isinstance(scene, dict):
        return ""
    return scene.get("voiceText") or scene.get("voice_text") or ""


def _srt_time(seconds):
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return "%02d:%02d:%02d,%03d" % (h, m, s, ms)


# libass renders `force_style` sizes against a PlayResY of 288 unless the subtitle file
# declares one, and an SRT cannot. So a FontSize is NOT pixels — it is scaled by
# height/288 on the way to the screen. Ignoring that scaled the size twice: 30 asked for
# on a 704-line frame arrived as ~73px and buried the picture. Everything user-facing here
# stays in PIXELS and converts at the last moment.
LIBASS_REF_H = 288


def to_libass(pixels, height):
    """Pixels on screen -> the FontSize/MarginV number libass needs to produce them."""
    return max(6, int(round(pixels * LIBASS_REF_H / max(int(height), 1))))


def auto_font_size(height):
    """Caption size as a fraction of frame height, not a fixed number of points.

    libass measures FontSize against the video's own pixel height, so one fixed value
    cannot serve every resolution: 24 is unobtrusive on 1080p and enormous on the 352-line
    preview renders — there it was 7% of the frame per line, and two lines buried the
    house behind the words. Broadcast captions sit near 4% of frame height, which is
    legible at every size this pipeline produces.
    """
    return max(10, int(round(height * 0.042)))


def wrap_width(width, font_size):
    """Characters per line that fill about 88% of the frame at this font size.

    Wrapping at a fixed 42 characters is the other half of the same bug: with the font
    scaled down, 42 characters no longer reach the edge, and with it scaled up they run
    past it. A sans-serif glyph averages close to half its point size in width.
    """
    usable = width * 0.88
    return max(20, min(60, int(usable / max(font_size * 0.5, 1))))


def _lines(text, width=42):
    """Word-wrap to `width`, returning every line — nothing dropped."""
    words, lines, line = str(text).split(), [], ""
    for word in words:
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = (line + " " + word).strip()
    if line:
        lines.append(line)
    return lines


def _cue_texts(text, width=42, per_cue=2):
    """Split narration into successive cues of at most `per_cue` lines each.

    Three lines of text would cover the frame, so a cue shows two — but the rest must
    become the NEXT cue, not disappear. Capping at two lines and discarding the overflow
    silently truncated every sentence the model wrote: a 30-word line of narration was
    subtitled as "...the broad frontage and crisp" and simply stopped, while the voice
    track carried on saying the whole thing.
    """
    lines = _lines(text, width)
    if not lines:
        return []
    return ["\n".join(lines[i:i + per_cue]) for i in range(0, len(lines), per_cue)]


def _split_duration(texts, duration):
    """Share a clip's runtime between its cues, in proportion to how much is said."""
    total = sum(len(t) for t in texts) or 1
    spans, clock = [], 0.0
    for index, text in enumerate(texts):
        span = duration * len(text) / total
        # The last cue absorbs any rounding so the cues always end exactly on the clip.
        end = duration if index == len(texts) - 1 else clock + span
        spans.append((clock, end))
        clock = end
    return spans


def _run(cmd, cwd=None, timeout=1800):
    """Run ffmpeg and raise on failure.

    `stdin=DEVNULL` is not tidiness, it is the fix for a hang that cost most of an hour.
    ffmpeg polls stdin for its interactive keys ('q' to quit). Launched from a node it
    inherits ComfyUI's console handle, and on Windows that poll can spin forever: the
    encode finishes, the output file is left a few KB short, and the process sits at
    ~20% of a core with nothing to read. Eleven clips went through in nine seconds and
    the twelfth hung for ten minutes; the identical command run by hand took 2.8 s.

    The timeout is the backstop. `subprocess.run` with no timeout waits for a stuck
    child forever, and a wedged ffmpeg then wedges the whole ComfyUI queue with no error
    and nothing in the log — which is exactly how the hang above presented.
    """
    try:
        proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "ffmpeg did not finish within %ds and was killed: %s\n"
            "Check for another process holding the file, or run the command by hand."
            % (timeout, " ".join(str(c) for c in cmd[:6]) + " ..."))
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
                "caption_style": ("ARK_CAPTION_STYLE", {
                    "tooltip": "Wire a Caption Style node here to choose the font and "
                               "one of the five subtitle styles. Leave it unconnected "
                               "for plain white captions at subtitle_size.",
                }),
                "subtitle_size": ("INT", {
                    "default": 0, "min": 0, "max": 200,
                    "tooltip": "0 = scale to the video (about 4% of frame height), which "
                               "is what you want when one workflow renders previews and "
                               "finals at different sizes. Only used when caption_style "
                               "is NOT connected — the style node carries its own size.",
                }),
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
            caption_style=None, subtitle_size=None, crf=None):
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
        # Typing "myfilm.mp4" is the obvious thing to do, and this node appends the
        # extension itself — leaving it produced `myfilm.mp4.mp4`.
        stem, ext = os.path.splitext(name)
        if stem and ext.lower() in (".mp4", ".mov", ".mkv", ".webm", ".m4v"):
            name = stem
        plan = _one(scenes_json, "") or ""
        music = str(_one(music_path, "") or "").strip()
        volume = _one(music_volume, 0.18)
        level = _one(normalize_speech, True)
        subs_on = _one(burn_subtitles, True)
        subs_size = int(_one(subtitle_size, 0) or 0)
        style = _one(caption_style, None)
        if style is not None and not style.get("enabled", True):
            # The Caption Style node's own off switch. Kept separate from
            # `burn_subtitles` so either one can silence captions on its own.
            subs_on, style = False, None
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
            paths, durations, speeches = [], [], []
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
                speeches.append(probe_speech(clip, durations[-1]))

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

            resolution = probe_size(joined) if subs_on and plan else (1280, 720)
            if subs_size <= 0:
                subs_size = auto_font_size(resolution[1])
            wrap = wrap_width(resolution[0], subs_size)

            subs = None
            if subs_on and plan:
                subs = (self._write_ass(work, plan, durations, style, resolution,
                                        speeches)
                        if style else self._write_srt(work, plan, durations, wrap,
                                                      speeches))

            final = os.path.join(folder, "%s.mp4" % name)
            cmd = [ffmpeg, "-y", "-i", joined]
            filters, maps = [], []
            if music and os.path.exists(music):
                cmd += ["-i", music]
                filters.append(
                    "[1:a]volume=%s[bed];"
                    "[0:a][bed]amix=inputs=2:duration=first:dropout_transition=0[aout]"
                    % volume)
                maps = ["-map", "0:v" if not subs else "[vout]", "-map", "[aout]"]
            elif music:
                print("[arkennemasis] music file not found, skipping bed: %s" % music)

            if subs:
                # Bare filename only — see SUBS_NAME above. ffmpeg runs with cwd=work.
                if style:
                    # The ASS carries the whole look in its own [V4+ Styles] block, so
                    # force_style must NOT be set here — it would override the lot.
                    spec = "subtitles=%s%s" % (SUBS_ASS, ass.fontsdir_arg(style))
                else:
                    # MarginV is in the same pixel space as FontSize, so it has to scale
                    # with the frame as well — a fixed 28 sat a third of the way up a
                    # 352-line render.
                    spec = ("subtitles=%s:force_style='FontSize=%d,"
                            "PrimaryColour=&Hffffff,OutlineColour=&H80000000,"
                            "BorderStyle=3,Outline=1,Shadow=0,Alignment=2,MarginV=%d'"
                            % (SUBS_NAME,
                               to_libass(subs_size, resolution[1]),
                               to_libass(max(8, int(round(resolution[1] * 0.05))),
                                         resolution[1])))
                filters.append("[0:v]%s[vout]" % spec)
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

    @staticmethod
    def _scenes(scenes_json):
        try:
            scenes = json.loads(scenes_json)
        except ValueError:
            print("[arkennemasis] scenes_json unparseable — skipping subtitles")
            return None
        if isinstance(scenes, dict):
            scenes = scenes.get("scenes") or scenes.get("output") or []
        return scenes if isinstance(scenes, list) and scenes else None

    def _write_ass(self, work, scenes_json, durations, style, resolution, speeches=None):
        """Subtitles as ASS, so the Caption Style node's choices actually reach libass."""
        scenes = self._scenes(scenes_json)
        if not scenes:
            return None

        needs_words = style.get("style") in ass.NEEDS_WORD_TIMING
        cues, clock = [], 0.0
        spoken = speeches or durations
        for scene, duration, speech in zip(scenes, durations, spoken):
            text = _voice_text(scene)
            if text and duration > 0:
                # Same split as the SRT path, so the styled captions and the plain ones
                # show the same words — a long line is several cues, never a truncation.
                texts = _cue_texts(text)
                for body, (start, end) in zip(texts, _split_duration(texts, speech)):
                    cue = {"text": body, "start": clock + start, "end": clock + end}
                    if needs_words:
                        span = end - start
                        # Inset the speech window, but never let it collapse on a
                        # short cue.
                        lead = min(LEAD_IN, span * 0.15)
                        tail = min(TAIL, span * 0.2)
                        cue["words"] = ass.estimate_words(
                            body, clock + start + lead, clock + end - tail)
                    cues.append(cue)
            clock += duration
        if not cues:
            return None

        path = os.path.join(work, SUBS_ASS)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(ass.build(cues, style, resolution))
        print("[arkennemasis] %d subtitle cues as ASS — %s in %s at %dx%d%s"
              % (len(cues), style.get("style"), style.get("font"),
                 resolution[0], resolution[1],
                 ", word timings estimated from the script" if needs_words else ""))
        return path

    def _write_srt(self, work, scenes_json, durations, wrap=42, speeches=None):
        scenes = self._scenes(scenes_json)
        if not scenes:
            return None

        lines, clock, cue = [], 0.0, 1
        # Positional pairing: clip i is scene i, because the loop preserves order.
        spoken = speeches or durations
        for scene, duration, speech in zip(scenes, durations, spoken):
            text = _voice_text(scene)
            if text and duration > 0:
                texts = _cue_texts(text, wrap)
                for body, (start, end) in zip(texts, _split_duration(texts, speech)):
                    lines.append("%d\n%s --> %s\n%s\n"
                                 % (cue, _srt_time(clock + start),
                                    _srt_time(clock + end), body))
                    cue += 1
            clock += duration
        if not lines:
            return None
        path = os.path.join(work, SUBS_NAME)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines))
        print("[arkennemasis] %d subtitle cues from the scene plan, wrapped at %d chars"
              % (cue - 1, wrap))
        return path


NODE_CLASS_MAPPINGS = {
    "ArkVideoAssemble": ArkVideoAssemble,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkVideoAssemble": "arkennemasis Video Assemble (clips + music + subs)",
}
