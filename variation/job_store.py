"""Stage 8 state — durable job records, resume, and the run report.

A node graph executes; it does not remember. At cell 380 of 720 a node throws, and a
graph holds no answer to "which cells are done, which passed, which need retry" — the
run restarts and 380 images are paid for twice. So the memory lives on disk, one JSON
file per cell, written through these nodes and readable by anything else: the
orchestrator, a later run, or a person with a text editor.

Three nodes:

    ArkJobSkip     resume. Its `fresh` input is LAZY, so a cell that already passed
                   never evaluates the generation branch at all — no API call, no
                   model load. The master is read back from disk instead.
    ArkJobRecord   the durable write: one attempt appended, status set.
    ArkRunReport   the §7.7 summary, and the first-time pass rate that is this
                   system's primary health metric.

The lazy skip is the same mechanism the pack's image-engine switch uses, and it is what
makes a re-run of a 700-cell product cost nothing for the cells that are already done.
Blocking the branch after the fact would still have paid for it.
"""

from __future__ import annotations

import datetime
import json
import os
import re

import numpy as np
import torch

from .schema import ValidationError

STATUSES = ("pending", "generating", "passed", "flagged", "approved", "delivered")


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def resolve_dir(path, fallback="variation/jobs"):
    path = str(path or "").strip() or fallback
    if not os.path.isabs(path):
        try:
            import folder_paths
            path = os.path.join(folder_paths.get_output_directory(), path)
        except Exception:
            path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)
    return path


def safe_key(key):
    """A cell key as a filename. The key itself stays intact inside the record."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(key or "job"))[:180] or "job"


def job_path(jobs_dir, key):
    return os.path.join(jobs_dir, "%s.json" % safe_key(key))


def load_job(jobs_dir, key):
    path = job_path(jobs_dir, key)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def save_job(jobs_dir, job):
    """Atomic write — a crash mid-save must never leave a half-parsed record."""
    path = job_path(jobs_dir, job.get("key"))
    temporary = path + ".part"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(job, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)
    return path


def get_or_create(jobs_dir, cell, recipe_hash="", prompt_hash=""):
    job = load_job(jobs_dir, cell.get("key"))
    if job is None:
        job = {
            "key": cell.get("key"),
            "filename": cell.get("filename"),
            "product": cell.get("product"),
            "plate": cell.get("plate"),
            "axes": cell.get("axes") or {},
            "recipe_hash": recipe_hash,
            "prompt_hash": prompt_hash,
            "status": "pending",
            "attempts": [],
            "master": None,
            "delivered": {},
            "created_at": _now(),
            "updated_at": _now(),
        }
    else:
        # A changed recipe means the existing master was made to different instructions.
        # Recording that here is what lets the report explain why a cell re-ran.
        if recipe_hash and job.get("recipe_hash") and job["recipe_hash"] != recipe_hash:
            job["recipe_changed_from"] = job["recipe_hash"]
            job["recipe_hash"] = recipe_hash
            if job.get("status") in ("passed", "approved", "delivered"):
                job["status"] = "pending"
        if prompt_hash:
            job["prompt_hash"] = prompt_hash
    return job


def _load_image(path):
    from PIL import Image
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(array)[None, ...]


class ArkJobSkip:
    CATEGORY = "arkennemasis/Variation"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("image", "job_json", "was_generated", "status")
    DESCRIPTION = (
        "Resume. If this cell already passed, its master is read from disk and the "
        "generation branch is NEVER evaluated — no API call, no model load. Wire the "
        "generated image into 'fresh'; it is lazy, so it only runs when the cell "
        "actually needs it."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cell_json": ("STRING", {
                    "multiline": True, "default": "{}", "forceInput": True,
                }),
                "jobs_dir": ("STRING", {
                    "default": "variation/jobs",
                    "tooltip": "Durable job records — one JSON per cell. Must survive "
                               "the run: never point this at a per-run folder.",
                }),
            },
            "optional": {
                "fresh": ("IMAGE", {
                    "lazy": True,
                    "tooltip": "The freshly generated + verified image. LAZY: skipped "
                               "entirely when this cell already passed.",
                }),
                "recipe_json": ("STRING", {
                    "multiline": True, "default": "", "forceInput": True,
                    "tooltip": "Optional. Supplies the recipe hash, so a recipe change "
                               "correctly invalidates cells made under the old one.",
                }),
                "force_rerun": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Regenerate even if the cell already passed. Off is the "
                               "point of this node.",
                }),
                "redo_flagged": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Re-attempt cells previously marked flagged. Off leaves "
                               "them alone so a sweep only fills genuine gaps.",
                }),
            },
        }

    def _should_run(self, cell_json, jobs_dir, recipe_json="", force_rerun=False,
                    redo_flagged=True):
        try:
            cell = json.loads(cell_json or "{}")
        except ValueError:
            return True, None, None
        recipe_hash = ""
        if str(recipe_json or "").strip():
            try:
                recipe_hash = (json.loads(recipe_json) or {}).get("recipe_hash") or ""
            except ValueError:
                recipe_hash = ""
        folder = resolve_dir(jobs_dir)
        job = get_or_create(folder, cell, recipe_hash)

        if force_rerun:
            return True, job, folder
        status = job.get("status")
        master = job.get("master")
        if status in ("passed", "approved", "delivered") and master and os.path.isfile(master):
            return False, job, folder
        if status == "flagged" and not redo_flagged:
            return False, job, folder
        return True, job, folder

    def check_lazy_status(self, cell_json, jobs_dir, fresh=None, recipe_json="",
                          force_rerun=False, redo_flagged=True):
        # Naming 'fresh' is what causes it to be evaluated. Not naming it leaves the
        # whole generation branch unevaluated, which is the entire saving.
        needed, _job, _folder = self._should_run(
            cell_json, jobs_dir, recipe_json, force_rerun, redo_flagged)
        return ["fresh"] if needed else []

    def run(self, cell_json, jobs_dir, fresh=None, recipe_json="", force_rerun=False,
            redo_flagged=True):
        needed, job, folder = self._should_run(
            cell_json, jobs_dir, recipe_json, force_rerun, redo_flagged)
        if job is None:
            raise ValidationError("ArkJobSkip: cell_json is not valid JSON.")

        if not needed:
            master = job.get("master")
            if master and os.path.isfile(master):
                print("[arkennemasis] skip %s — already %s" % (job["key"], job["status"]))
                return (_load_image(master), json.dumps(job, ensure_ascii=False),
                        False, job.get("status", "passed"))
            # Flagged with no master: nothing to hand on, but the run must not halt.
            print("[arkennemasis] skip %s — %s, no master on disk"
                  % (job["key"], job.get("status")))
            return (torch.zeros((1, 8, 8, 3), dtype=torch.float32),
                    json.dumps(job, ensure_ascii=False), False,
                    job.get("status", "flagged"))

        if fresh is None:
            raise ValidationError(
                "ArkJobSkip: this cell needs generating but nothing is wired into "
                "'fresh'.")
        job["status"] = "generating"
        job["updated_at"] = _now()
        save_job(folder, job)
        return (fresh, json.dumps(job, ensure_ascii=False), True, "generating")


class ArkJobRecord:
    CATEGORY = "arkennemasis/Variation"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("job_json", "status", "summary")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Write this cell's outcome to durable storage: the attempt with its "
        "measurements, the resulting status, and the master's path. This is the write "
        "that makes a run resumable."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cell_json": ("STRING", {
                    "multiline": True, "default": "{}", "forceInput": True,
                }),
                "jobs_dir": ("STRING", {"default": "variation/jobs"}),
                "verdict_json": ("STRING", {
                    "multiline": True, "default": "{}", "forceInput": True,
                    "tooltip": "From Verify Candidate — the measurements and the "
                               "pass/soft/hard result.",
                }),
            },
            "optional": {
                "master_path": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "Where the accepted image was written.",
                }),
                "recipe_json": ("STRING", {
                    "multiline": True, "default": "", "forceInput": True,
                }),
                "prompt_hash": ("STRING", {"default": "", "forceInput": True}),
                "qc_diagnosis": ("STRING", {
                    "multiline": True, "default": "", "forceInput": True,
                    "tooltip": "The critic's explanation. Recorded on the attempt, so a "
                               "flagged cell says WHY it was flagged rather than only "
                               "that it was.",
                }),
                "qc_passed": ("BOOLEAN", {"default": True, "forceInput": True}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, cell_json, jobs_dir, verdict_json, master_path="", recipe_json="",
            prompt_hash="", qc_diagnosis="", qc_passed=True):
        cell = json.loads(cell_json or "{}")
        verdict = json.loads(verdict_json or "{}")
        recipe_hash = ""
        if str(recipe_json or "").strip():
            recipe_hash = (json.loads(recipe_json) or {}).get("recipe_hash") or ""

        folder = resolve_dir(jobs_dir)
        job = get_or_create(folder, cell, recipe_hash, str(prompt_hash or ""))

        result = str(verdict.get("result") or "unknown")
        attempt = {
            "at": _now(),
            "result": result,
            "checks": verdict.get("checks") or {},
            "failures": verdict.get("failures") or [],
        }
        if str(qc_diagnosis or "").strip():
            attempt["qc"] = {"passed": bool(qc_passed),
                             "diagnosis": str(qc_diagnosis).strip()}
            # A critic that rejects a cell the arithmetic accepted is the entire reason
            # it exists, so its verdict has to be able to override a numeric pass.
            if not qc_passed and result == "pass":
                result = "soft"
                attempt["result"] = result
                attempt["failures"] = (attempt["failures"] or []) + [
                    "QC critic rejected it: " + str(qc_diagnosis).strip()[:200]]
        job.setdefault("attempts", []).append(attempt)

        if result == "pass":
            # This node and ArkDeliver are independent OUTPUT nodes, and ComfyUI does
            # not fix their relative order. Writing "passed" unconditionally meant that
            # whichever ran second undid the other's work: record-then-deliver ended at
            # "delivered", deliver-then-record ended back at "passed", and the terminal
            # status of a cell depended on scheduling. Derive it from whether delivery
            # has actually happened instead, so both orders converge.
            job["status"] = "delivered" if job.get("delivered") else "passed"
            if str(master_path or "").strip():
                job["master"] = str(master_path).strip()
        elif result == "hard":
            # Do not retry blindly: repeated hard failure usually means a recipe error,
            # and retrying spends money without addressing the cause.
            job["status"] = "flagged"
        else:
            attempts = len(job["attempts"])
            limit = 3
            if str(recipe_json or "").strip():
                limit = int(((json.loads(recipe_json) or {}).get("verification") or {})
                            .get("max_retries", 3))
            job["status"] = "flagged" if attempts > limit else "pending"

        job["updated_at"] = _now()
        path = save_job(folder, job)

        summary = "\n".join([
            "%s  ->  %s" % (job["key"], job["status"].upper()),
            "  attempts : %d" % len(job["attempts"]),
            "  result   : %s" % result,
            "  master   : %s" % (job.get("master") or "-"),
            "  record   : %s" % path,
        ] + ["  %-22s %s" % (k, v) for k, v in sorted((verdict.get("checks") or {}).items())])

        print("[arkennemasis] job %s -> %s (attempt %d)"
              % (job["key"], job["status"], len(job["attempts"])))
        return (json.dumps(job, ensure_ascii=False), job["status"], summary)


class ArkRunReport:
    CATEGORY = "arkennemasis/Variation"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("report", "first_time_pass_rate", "flagged_count", "flagged_keys")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Summarise the run: cells attempted, passed first time, passed after retry, "
        "flagged, and total generation calls. First-time pass rate is the system's "
        "primary health metric and the number worth quoting commercially."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "jobs_dir": ("STRING", {"default": "variation/jobs"}),
            },
            "optional": {
                "product": ("STRING", {
                    "default": "",
                    "tooltip": "Only count jobs for this product. Blank = every job in "
                               "the folder.",
                }),
                "write_to": ("STRING", {
                    "default": "",
                    "tooltip": "Optional folder to write report.txt and report.json.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, jobs_dir, product="", write_to=""):
        folder = resolve_dir(jobs_dir)
        wanted = str(product or "").strip()

        jobs = []
        for entry in sorted(os.listdir(folder)):
            if not entry.endswith(".json"):
                continue
            try:
                with open(os.path.join(folder, entry), "r", encoding="utf-8") as handle:
                    job = json.load(handle)
            except (OSError, ValueError):
                continue
            if wanted and job.get("product") != wanted:
                continue
            jobs.append(job)

        total = len(jobs)
        by_status = {}
        first_time = after_retry = calls = 0
        flagged_keys = []
        for job in jobs:
            status = job.get("status", "pending")
            by_status[status] = by_status.get(status, 0) + 1
            attempts = job.get("attempts") or []
            calls += len(attempts)
            if status in ("passed", "approved", "delivered"):
                if len(attempts) <= 1:
                    first_time += 1
                else:
                    after_retry += 1
            if status == "flagged":
                flagged_keys.append(job.get("key"))

        rate = (first_time / total) if total else 0.0

        lines = [
            "RUN REPORT%s" % (" — %s" % wanted if wanted else ""),
            "  cells attempted    : %d" % total,
            "  passed first time  : %d  (%.1f%%)" % (first_time, rate * 100),
            "  passed after retry : %d" % after_retry,
            "  flagged            : %d" % len(flagged_keys),
            "  generation calls   : %d" % calls,
            "  calls per delivered: %.2f"
            % (calls / max(1, first_time + after_retry)),
            "",
            "BY STATUS",
        ]
        for status in STATUSES:
            if status in by_status:
                lines.append("  %-12s %d" % (status, by_status[status]))
        for status, count in sorted(by_status.items()):
            if status not in STATUSES:
                lines.append("  %-12s %d" % (status, count))

        if flagged_keys:
            lines += ["", "FLAGGED — these need an operator, not a retry"]
            for key in flagged_keys[:40]:
                lines.append("  %s" % key)
            if len(flagged_keys) > 40:
                lines.append("  ... and %d more" % (len(flagged_keys) - 40))

        report = "\n".join(lines)
        payload = {
            "product": wanted or None,
            "total": total,
            "first_time": first_time,
            "after_retry": after_retry,
            "flagged": len(flagged_keys),
            "calls": calls,
            "first_time_pass_rate": round(rate, 4),
            "by_status": by_status,
            "flagged_keys": flagged_keys,
            "generated_at": _now(),
        }

        destination = str(write_to or "").strip()
        if destination:
            destination = resolve_dir(destination)
            with open(os.path.join(destination, "report.txt"), "w",
                      encoding="utf-8") as handle:
                handle.write(report + "\n")
            with open(os.path.join(destination, "report.json"), "w",
                      encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)

        print("[arkennemasis] run report: %d cells, %.1f%% first-time pass, %d flagged"
              % (total, rate * 100, len(flagged_keys)))
        return (report, float(rate), len(flagged_keys), "\n".join(map(str, flagged_keys)))


NODE_CLASS_MAPPINGS = {
    "ArkJobSkip": ArkJobSkip,
    "ArkJobRecord": ArkJobRecord,
    "ArkRunReport": ArkRunReport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkJobSkip": "arkennemasis Job Skip (resume — never re-pay)",
    "ArkJobRecord": "arkennemasis Job Record (durable state)",
    "ArkRunReport": "arkennemasis Run Report (pass rate + flagged)",
}


class ArkRunCollect:
    CATEGORY = "arkennemasis/Variation"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("jobs_dir", "delivered", "report")
    INPUT_IS_LIST = True          # receive EVERY cell's result in one call
    DESCRIPTION = (
        "The barrier between generating and summarising. With a fan-out the cells run "
        "concurrently and finish in no particular order, so anything that reports on the "
        "whole run has to wait for all of them — otherwise the board and the store "
        "export are free to execute first and write nothing, while the run still "
        "reports success."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "delivered_paths": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Wire Deliver's master_path here. It arrives as the whole "
                               "list, which is what forces the wait.",
                }),
                "jobs_dir": ("STRING", {"default": "variation/jobs"}),
            },
            "optional": {
                # Waiting on Deliver alone was not a barrier at all for the thing that
                # actually matters. ArkJobRecord is a SEPARATE output node writing the
                # durable status, so the report, board and store export could all read
                # a job record before its status had been written — reporting a cell as
                # unfinished purely because the scheduler had not reached it yet.
                # Taking Job Record's report as a second list input makes the wait
                # cover both writers.
                "job_reports": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Wire Remember this variation's report here too. Without "
                               "it the wait covers only the file writes, not the "
                               "durable status the summary reads.",
                }),
            },
        }

    def run(self, delivered_paths, jobs_dir, job_reports=None):
        # INPUT_IS_LIST makes every input a list, scalars included.
        folder = jobs_dir[0] if isinstance(jobs_dir, list) else jobs_dir
        paths = [p for p in (delivered_paths or []) if str(p or "").strip()]
        real = [p for p in paths if os.path.isfile(p)]
        recorded = len([r for r in (job_reports or []) if str(r or "").strip()])
        report = "\n".join([
            "RUN COLLECTED",
            "  cells delivered : %d" % len(paths),
            "  files on disk   : %d" % len(real),
            "  statuses written: %s" % (recorded if job_reports is not None
                                        else "not wired — see this node's tooltip"),
            "  jobs            : %s" % folder,
            "",
            "Everything downstream of this node is guaranteed to see the finished run.",
        ])
        if job_reports is not None and recorded != len(paths):
            # Not fatal — the summary reads the job files, not this count — but a
            # mismatch means one of the two writers did not run for every cell, and
            # that is worth seeing in the report rather than inferring from a total.
            print("[arkennemasis] collect: %d delivered but %d status record(s) — the "
                  "summary may under-report" % (len(paths), recorded))
        print("[arkennemasis] collected %d delivered cell(s)" % len(real))
        return (folder, len(real), report)


NODE_CLASS_MAPPINGS["ArkRunCollect"] = ArkRunCollect
NODE_DISPLAY_NAME_MAPPINGS["ArkRunCollect"] = \
    "arkennemasis Run Collect (wait for every cell)"
