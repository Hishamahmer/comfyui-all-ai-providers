"""A local run log — the state layer, without a spreadsheet.

Storyboard pipelines often carry a spreadsheet as their memory: one row per
scene, columns filled in as each stage completes, and a status column that says what
still needs doing. That works, but it puts an OAuth login and a network round trip in
the middle of a local render, and it leaks the whole project into someone's Drive.

This does the same job with a JSON file inside the run folder. Rows are upserted by
``row_key``, so a stage that reruns updates its row instead of appending a duplicate,
and a partial run leaves a log that says exactly which scenes are done.

Shape on disk — a dict, not a list, so an interrupted write can never scramble the
ordering and a lookup by key is O(1)::

    {"rows": {"1": {"scene_number": 1, "image_prompt": "...", "generation_status": "done"},
              "2": {...}},
     "updated": "2026-08-05T01:22:03"}

Wire ``folder_path`` from ``ArkRunFolder`` so one run keeps one log. Wire the branch's
image (or any upstream output) into ``gate`` and a skipped branch logs nothing, the same
way ``ArkTextFileSave`` avoids orphan captions.
"""

import json
import os


def resolve_folder(folder_path):
    """Absolute folder, defaulting relative paths into ComfyUI's output directory.

    Shared with ArkTextFileSave's behaviour on purpose: a node that writes beside an
    image must resolve paths the same way that image save does, or the pair splits.
    """
    folder = str(folder_path).strip()
    if not folder or not os.path.isabs(folder):
        try:
            import folder_paths
            base = folder_paths.get_output_directory()
        except Exception:
            base = os.getcwd()
        folder = os.path.join(base, folder) if folder else base
    os.makedirs(folder, exist_ok=True)
    return folder


class ArkRunLog:
    CATEGORY = "arkennemasis/Utility"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("log_path", "row_json")
    OUTPUT_NODE = True
    DESCRIPTION = ("Upsert one row into a local JSON run log — the offline replacement "
                   "for a Google Sheet state layer. Keyed by row_key, so reruns update "
                   "rather than duplicate.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "",
                    "tooltip": "Wire this from Run Folder so one run keeps one log. "
                               "Relative paths resolve inside ComfyUI's output dir.",
                }),
                "row_key": ("STRING", {
                    "default": "1",
                    "tooltip": "Identifies the row — the scene number in a scene "
                               "pipeline. Writing the same key again UPDATES that row.",
                }),
                "fields_json": ("STRING", {
                    "multiline": True, "default": "{}",
                    "tooltip": "A JSON object of the columns to write, e.g. "
                               '{"image_prompt": "...", "generation_status": "done"}. '
                               "Merged into any existing row; omitted keys are kept.",
                }),
            },
            "optional": {
                "log_name": ("STRING", {
                    "default": "run_log",
                    "tooltip": "Filename stem, no extension. Written as <stem>.json.",
                }),
                "gate": ("IMAGE", {
                    "tooltip": "Optional. Wire the branch's image in and a branch "
                               "skipped by a gate logs nothing — no row claiming work "
                               "that never happened.",
                }),
                "merge": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "On = merge into the existing row (normal). Off = replace "
                               "that row outright.",
                }),
                "row_number": ("INT", {
                    "default": 0, "min": 0, "max": 100000,
                    "tooltip": "Overrides row_key when above 0. Wire the scene number "
                               "here: inside a per-scene loop the node runs once per "
                               "iteration, and a fixed row_key would make every "
                               "iteration overwrite the same row.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # A log write must happen every run. Without this, ComfyUI's cache would serve
        # the previous result and the second run would silently record nothing.
        return float("nan")

    def run(self, folder_path, row_key, fields_json, log_name="run_log", gate=None,
            merge=True, row_number=0):
        if row_number:
            row_key = str(int(row_number))
        try:
            fields = json.loads(fields_json) if str(fields_json).strip() else {}
        except ValueError as exc:
            raise ValueError("ArkRunLog: fields_json is not valid JSON (%s). First 200 "
                             "chars: %s" % (exc, str(fields_json)[:200]))
        if not isinstance(fields, dict):
            raise ValueError("ArkRunLog: fields_json must be a JSON object, got %s."
                             % type(fields).__name__)

        folder = resolve_folder(folder_path)
        stem = os.path.basename(str(log_name).strip()) or "run_log"
        path = os.path.join(folder, "%s.json" % stem)

        data = {"rows": {}}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict) and isinstance(loaded.get("rows"), dict):
                    data = loaded
            except (ValueError, OSError) as exc:
                # Never lose a long run's history to one bad read: keep the damaged file
                # so it can be inspected, and start a fresh log beside it.
                broken = path + ".broken"
                try:
                    os.replace(path, broken)
                    print("[arkennemasis] run log unreadable (%s); moved to %s"
                          % (exc, broken))
                except OSError:
                    pass

        key = str(row_key).strip()
        row = dict(data["rows"].get(key, {})) if merge else {}
        row.update(fields)
        data["rows"][key] = row

        import datetime
        data["updated"] = datetime.datetime.now().isoformat(timespec="seconds")

        tmp = path + ".part"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)

        print("[arkennemasis] run log row '%s' -> %s (%d rows)"
              % (key, path, len(data["rows"])))
        return (path, json.dumps(row, ensure_ascii=False))


NODE_CLASS_MAPPINGS = {
    "ArkRunLog": ArkRunLog,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkRunLog": "arkennemasis Run Log (local, no spreadsheet)",
}
