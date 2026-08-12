"""One whole scene of Hailuo video in a single node — sample, decode, save, free.

WHY THIS EXISTS, and it is not a style preference.

ComfyUI's list execution runs **stage by stage, not item by item**. Given N scenes it
finishes the sampler for every scene, then decodes every scene, then saves every scene.
Every intermediate latent is therefore alive at once. Measured on this machine: three
608x352 clips totalling 8.4 seconds spent **15 of their 23 minutes** in the decode phase,
GPU at 31% with 47.9 of 51.5 GB held and `0 models unloaded` repeating — thrashing, not
computing. The same box had rendered two *heavier* 1280x736 clips in 254 seconds total
when each branch ran to completion before the next began.

Collapsing the whole per-scene chain into one node restores that behaviour while keeping
a single canvas. Each call samples, decodes both streams, muxes, writes the file, drops
every tensor and empties the cache — then returns only a path. Peak memory is one scene
no matter whether you render 5 or 50.

The trade is that the sampler wiring lives in here rather than on the canvas. The
workflow keeps a muted copy of the equivalent node chain as documentation of what this
does internally.
"""

import os

# LOCKED. Hailuo MiniMax H3 generates a frame COUNT at 24 fps — the length maths, the
# audio it produces and the subtitle timings all assume it. Muxing at anything else
# plays the picture at the wrong speed against its own soundtrack and desyncs the lips.
# Deliberately not a widget: a value that must never change should not be a box that can
# be emptied.
FPS = 24.0


def _first(result):
    """Unwrap a v3 io.NodeOutput (or a plain tuple) to its first value."""
    args = getattr(result, "args", None)
    if args is not None:
        return args[0]
    return result[0] if isinstance(result, (tuple, list)) else result


def _both(result):
    args = getattr(result, "args", None)
    if args is not None:
        return args[0], args[1]
    return result[0], result[1]



def _release_everything(when):
    """Evict models and the pinned host buffers behind them, and say what it recovered.

    `unload_all_models` is the call that actually evicts; without it ComfyUI reports
    "0 models unloaded" and keeps ~94 GB of staged weights (63 GB model + 26 GB text
    encoder + 5 GB VAE) in host RAM. Freeing VRAM alone is not enough — that host side
    is what runs out, and the failure surfaces as `HostBuffer.read_file_slice failed`
    or a CUDA OOM raised while the GPU is nearly empty.

    Every call here is optional across ComfyUI versions, so none may fail hard.
    """
    import gc

    import comfy.model_management as mm

    before_vram = mm.get_free_memory(mm.get_torch_device()) / 1e9
    before_ram = _free_ram_gb()

    gc.collect()
    try:
        mm.unload_all_models()
    except Exception as exc:
        print("[arkennemasis] unload_all_models failed (%s)" % str(exc)[:80])
    for name in ("cleanup_models_gc", "cleanup_models", "reset_cast_buffers"):
        fn = getattr(mm, name, None)
        if callable(fn):
            try:
                fn()
            except Exception as exc:
                print("[arkennemasis] %s skipped (%s)" % (name, str(exc)[:70]))
    try:
        import comfy_aimdo.host_buffer as hb
        cleanup = getattr(hb, "cleanup_file_reader", None)
        if callable(cleanup):
            cleanup()
    except Exception:
        pass
    try:
        mm.soft_empty_cache(force=True)
    except Exception:
        pass
    gc.collect()

    after_vram = mm.get_free_memory(mm.get_torch_device()) / 1e9
    after_ram = _free_ram_gb()
    ram_note = ""
    if before_ram is not None and after_ram is not None:
        ram_note = " | RAM %.1f -> %.1f GB" % (before_ram, after_ram)
    print("[arkennemasis] release (%s): VRAM %.1f -> %.1f GB%s"
          % (when, before_vram, after_vram, ram_note))


def _free_ram_gb():
    """Free system RAM. The number that actually decides whether the next scene runs."""
    try:
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        s = Status()
        s.dwLength = ctypes.sizeof(Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
            return s.ullAvailPhys / 1e9
    except Exception:
        pass
    return None


class ArkHailuoScene:
    CATEGORY = "arkennemasis/Video"
    FUNCTION = "run"
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "path")
    DESCRIPTION = ("Render ONE scene end to end with Hailuo MiniMax H3 — condition, "
                   "sample, decode video+audio, mux and write the file — then free "
                   "everything. Inside a per-scene loop this keeps peak memory at one "
                   "scene instead of holding every scene's latents at once.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE", {"tooltip": "The VIDEO vae."}),
                "audio_vae": ("VAE", {"tooltip": "The AUDIO vae — this is what gives "
                                                 "the clip its soundtrack."}),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "image": ("IMAGE", {"tooltip": "This scene's still — the FIRST frame of "
                                               "the clip."}),
                "prompt": ("STRING", {"multiline": True, "default": "",
                                      "forceInput": True}),
                "width": ("INT", {"default": 1280, "min": 32, "max": 16384, "step": 32}),
                "height": ("INT", {"default": 736, "min": 32, "max": 16384, "step": 32}),
                "length": ("INT", {
                    "default": 243, "min": 5, "max": 2048, "forceInput": True,
                    "tooltip": "Frames for THIS scene, wired from Scene List so each "
                               "clip runs its own duration. forceInput on purpose: a "
                               "widget carrying a link occupies a widgets_values slot "
                               "and any disagreement about slot counts shifts every "
                               "later value.",
                }),
                "seed": ("INT", {
                    "default": 1000, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Base seed. It is mixed with THIS scene's prompt, so "
                               "every scene of a run samples differently instead of all "
                               "of them sharing one noise pattern.",
                }),
                "reseed_each_run": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Draw fresh noise on every run. With this off, re-running "
                               "an unchanged scene reproduces the SAME clip down to the "
                               "audio - including the same garbled speech, which looks "
                               "like the scene is locked. Turn it off only when you want "
                               "to reproduce a specific take exactly.",
                }),
                "filename_prefix": ("STRING", {
                    "default": "nemasis/scene", "forceInput": True,
                    "tooltip": "Relative to ComfyUI's output dir. Each call appends the "
                               "next counter, so clips land in scene order.",
                }),
            },
            "optional": {
                # Optional, and added AFTER the required block, so every graph built
                # before this existed keeps working untouched: H3 already accepted a
                # closing frame, this node just never offered one.
                "last_frame": ("IMAGE", {
                    "tooltip": "Optional CLOSING frame. Given one, H3 animates the "
                               "transition from `image` to this, instead of inventing "
                               "motion from a single still. This is what a "
                               "first-frame-to-last-frame transformation needs.",
                }),
            },
        }

    def run(self, model, clip, vae, audio_vae, sampler, sigmas, image, prompt,
            width, height, length, seed, filename_prefix,
            reseed_each_run=True, last_frame=None):
        fps = FPS
        import torch
        import folder_paths
        import comfy.model_management as mm
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo
        from comfy_extras.nodes_custom_sampler import (
            BasicGuider, RandomNoise, SamplerCustomAdvanced,
        )
        from comfy_extras.nodes_audio import VAEDecodeAudio
        from comfy_api.latest._util.video_types import VideoComponents
        from comfy_api.latest._input_impl.video_types import (
            VideoFromComponents, VideoFromFile,
        )
        from fractions import Fraction
        import nodes as comfy_nodes

        length = int(length)
        print("[arkennemasis] scene: %dx%d, %d frames (%.2fs)"
              % (width, height, length, length / max(fps, 1e-6)))

        # Free BEFORE staging, not only after. Freeing at the end is not enough on its
        # own — and a separate purge NODE cannot help, because `ArkSceneList` declares
        # OUTPUT_IS_LIST, so ComfyUI runs each node across every scene before moving to
        # the next one. Every purge in the graph therefore fires up front, when nothing
        # is loaded, and never between shots. Measured: three purges reporting
        # "VRAM 49.8 -> 49.8 GB (+0.0)", then shot 1 ending on 1.3 GB VRAM / 5.7 GB RAM
        # free, then the process dying while staging the text encoder for shot 2.
        #
        # This is the only code that runs between one shot and the next, so the release
        # belongs here. `unload_all_models` is the call that actually evicts — the
        # cleanup at the end of this function reports "0 models unloaded" without it.
        _release_everything("before staging")

        # --- condition + empty latent -------------------------------------
        if last_frame is not None:
            print("[arkennemasis] scene: last_frame supplied — rendering a "
                  "first-frame-to-last-frame transition")
        cond, latent = _both(MiniMaxH3ImageToVideo.execute(
            clip=clip, vae=vae, prompt=prompt, width=int(width), height=int(height),
            length=length, first_frame=image, last_frame=last_frame))

        # --- sample --------------------------------------------------------
        guider = _first(BasicGuider.execute(model=model, conditioning=cond))
        # One fixed seed for every scene AND every run meant two things: all N clips
        # sampled from identical noise, and re-rendering reproduced a bad take exactly —
        # same garbled speech every time, which reads as "the scene is locked".
        # Mixing the prompt in makes scenes differ from each other deterministically;
        # reseed_each_run makes a re-run a genuinely new take.
        import hashlib
        mixed = int(hashlib.sha256(
            ("%d|%s" % (int(seed), prompt)).encode("utf-8")).hexdigest()[:15], 16)
        if reseed_each_run:
            import random
            mixed ^= random.getrandbits(48)
        mixed &= 0xffffffffffffffff
        print("[arkennemasis] scene seed %d (base %d%s)"
              % (mixed, int(seed), ", reseeded" if reseed_each_run else ", fixed"))
        noise = _first(RandomNoise.execute(noise_seed=mixed))
        samples = _first(SamplerCustomAdvanced.execute(
            noise=noise, guider=guider, sampler=sampler, sigmas=sigmas,
            latent_image=latent))
        del cond, latent, guider, noise

        # --- decode both streams from the SAME latent ----------------------
        # This is Hailuo's joint generation: picture and sound come out of one pass.
        images = _first(comfy_nodes.VAEDecode().decode(vae, samples))
        audio = _first(VAEDecodeAudio.execute(vae=audio_vae, samples=samples))
        del samples

        # --- mux and write --------------------------------------------------
        video = VideoFromComponents(VideoComponents(
            images=images, audio=audio, frame_rate=Fraction(round(float(fps) * 1000),
                                                            1000)))
        folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(),
            images.shape[2], images.shape[1])
        filename = "%s_%05d_.mp4" % (name, counter)
        path = os.path.join(folder, filename)
        video.save_to(path)
        del video, images, audio

        # Hand back a file-backed VIDEO: it holds a path, not frames, so the assembler
        # downstream can collect every scene without keeping any of them in memory.
        result = VideoFromFile(path)

        # The whole point of this node — release before the next scene starts.
        mm.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # VRAM was never the thing that ran out. ComfyUI's dynamic loading stages model
        # weights into PINNED HOST buffers and keeps them: at 1280x736 that is ~94 GB of
        # system RAM (63 GB model + 26 GB text encoder + 5 GB VAE). Freeing only VRAM let
        # that grow across scenes until scene 2 or 3 died with
        # "HostBuffer.read_file_slice failed" and a CUDA OOM raised while the GPU was
        # nearly empty (CUDA OOMs: 0, PyTorch holding 1.2 GB). These calls release the
        # host side. Each is optional across ComfyUI versions, so none may fail hard.
        # Same release as on entry — this one previously omitted `unload_all_models`,
        # which is why it reported "0 models unloaded" and freed nothing.
        _release_everything("after saving")

        free = mm.get_free_memory(mm.get_torch_device()) / 1e9
        ram = _free_ram_gb()
        print("[arkennemasis] scene saved -> %s  (%.1f GB VRAM free%s)"
              % (filename, free,
                 "" if ram is None else ", %.1f GB RAM free" % ram))

        return {"ui": {"images": [{"filename": filename, "subfolder": subfolder,
                                   "type": "output"}]},
                "result": (result, path)}


NODE_CLASS_MAPPINGS = {
    "ArkHailuoScene": ArkHailuoScene,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkHailuoScene": "arkennemasis Hailuo Scene (one clip, start to finish)",
}
