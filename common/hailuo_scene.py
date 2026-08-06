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
                "image": ("IMAGE", {"tooltip": "This scene's still — the first frame."}),
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
        }

    def run(self, model, clip, vae, audio_vae, sampler, sigmas, image, prompt,
            width, height, length, seed, filename_prefix,
            reseed_each_run=True):
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

        # --- condition + empty latent -------------------------------------
        cond, latent = _both(MiniMaxH3ImageToVideo.execute(
            clip=clip, vae=vae, prompt=prompt, width=int(width), height=int(height),
            length=length, first_frame=image, last_frame=None))

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

        free = mm.get_free_memory(mm.get_torch_device()) / 1e9
        print("[arkennemasis] scene saved -> %s  (%.1f GB VRAM free after cleanup)"
              % (filename, free))

        return {"ui": {"images": [{"filename": filename, "subfolder": subfolder,
                                   "type": "output"}]},
                "result": (result, path)}


NODE_CLASS_MAPPINGS = {
    "ArkHailuoScene": ArkHailuoScene,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkHailuoScene": "arkennemasis Hailuo Scene (one clip, start to finish)",
}
