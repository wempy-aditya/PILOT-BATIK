#!/usr/bin/env python3
"""
Batik Inpainting API - Precise Motif Editing
Edit specific areas while preserving the rest
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import logging
import os
import sys
import json
import datetime
from pathlib import Path
from PIL import Image, ImageDraw
from io import BytesIO
import random
import traceback
import numpy as np
import cv2
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
from peft import PeftModel
import gc

# -- Allow importing utils/evaluation.py from parent PILOT directory
_PILOT_ROOT = Path(__file__).resolve().parent.parent
if str(_PILOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PILOT_ROOT))

try:
    from utils.evaluation import (
        compute_clip_score,
        compute_nima_score,
        compute_ssim_non_mask,
        compute_lpips_non_mask,
    )
    _EVAL_AVAILABLE = True
except ImportError:
    _EVAL_AVAILABLE = False

# Directory where all results + eval JSON will be saved automatically
SAVE_DIR = _PILOT_ROOT / "outputs" / "our_experiments"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s %(message)s')
logger = logging.getLogger(__name__)

if not _EVAL_AVAILABLE:
    logger.warning("utils/evaluation.py not found -- /evaluate endpoint will be disabled.")

app = Flask(__name__)
CORS(app)


class BatikInpaintingEditor:
    """Inpainting editor for precise motif editing"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model = "runwayml/stable-diffusion-inpainting"
        self.pipe = None
        self.current_scenario = None
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        self.lora_paths = {
            "scenario1": "/home/if24/code/batikt2i/output/lora_weights/scenario1/final_lora/",
            "scenario2": "/home/if24/code/batikt2i/output/lora_weights/scenario2/final_lora/",
            "scenario2_1": "/home/if24/code/batikt2i/output/lora_weights/scenario2_1/final_lora/",
            "scenario2_2": "/home/if24/code/batikt2i/output/lora_weights/scenario2_2/final_lora/",
            "scenario2_3": "/home/if24/code/batikt2i/output/lora_weights/scenario2_3/final_lora/",
            "scenario2_4": "/home/if24/code/batikt2i/output/lora_weights/scenario2_4/final_lora/",
            "scenario2_5": "/home/if24/code/batikt2i/output/lora_weights/scenario2_5/final_lora/",
            "scenario3_1": "/home/if24/code/batikt2i/output/lora_weights/scenario3_1/final_lora/",
            "scenario3_2": "/home/if24/code/batikt2i/output/lora_weights/scenario3_2/final_lora/",
            "scenario4_1": "/home/if24/code/new_batik_t2i_train/output/lora_weights/scenario4_1_1/final_lora/",
            "scenario4_1_1": "/home/if24/code/new_batik_t2i_train/output/lora_weights/scenario4_1_1/final_lora/",
        }

    def create_mask_from_color(self, image, target_color, tolerance=30):
        img_array = np.array(image)
        color_diff = np.abs(img_array - np.array(target_color))
        mask = np.all(color_diff <= tolerance, axis=-1)
        return Image.fromarray((mask * 255).astype(np.uint8))

    def load_pipeline(self, scenario):
        if self.pipe is None or self.current_scenario != scenario:
            logger.info(f"Loading Inpainting pipeline for scenario: {scenario}")
            if self.pipe is not None:
                del self.pipe
                gc.collect()
                torch.cuda.empty_cache()
            logger.info("   Loading SD Inpainting pipeline...")
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )
            lora_path = self.lora_paths.get(scenario)
            if lora_path and Path(lora_path).exists():
                logger.info("   WARNING: Skipping LoRA (SD v1.5 LoRA incompatible with SD Inpainting 9ch UNet)")
            else:
                logger.info("   INFO: Using base SD Inpainting model")
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
            self.pipe = self.pipe.to(self.device)
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                logger.info("   OK Enabled xFormers")
            except:
                logger.info("   WARNING xFormers not available")
            self.current_scenario = scenario
            logger.info("   OK Inpainting pipeline loaded!")

    def _prepare_images(self, input_image, mask_image):
        orig_size = input_image.size
        input_resized = input_image.resize((512, 512), Image.LANCZOS)
        mask_resized = mask_image.resize((512, 512), Image.NEAREST)
        mask_array = np.array(mask_resized)
        kernel = np.ones((15, 15), np.uint8)
        mask_dilated = cv2.dilate(mask_array, kernel, iterations=1)
        mask_resized = Image.fromarray(mask_dilated)
        return input_resized, mask_resized, orig_size

    @torch.no_grad()
    def inpaint_image(self, input_image, mask_image, prompt, scenario,
                      negative_prompt="blurry, bad quality, distorted, ugly, deformed",
                      steps=30, guidance_scale=7.5, strength=0.99, seed=None):
        self.load_pipeline(scenario)
        if input_image.size != mask_image.size:
            mask_image = mask_image.resize(input_image.size, Image.NEAREST)
        if seed is None or seed == -1:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        orig_size = input_image.size
        logger.info(f"Inpainting with prompt: '{prompt[:60]}...'")
        logger.info(f"   Image size: {orig_size}")
        logger.info(f"   Steps: {steps}, CFG: {guidance_scale}, Seed: {seed}")

        mask_array = np.array(mask_image)
        rows = np.any(mask_array > 127, axis=1)
        cols = np.any(mask_array > 127, axis=0)
        if not rows.any():
            logger.warning("   WARNING Mask is empty, returning original image")
            return input_image, mask_image, seed

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        h_box = rmax - rmin
        w_box = cmax - cmin
        pad = max(32, int(max(h_box, w_box) * 0.20))
        img_w, img_h = input_image.size
        x1 = max(0, cmin - pad)
        y1 = max(0, rmin - pad)
        x2 = min(img_w, cmax + pad)
        y2 = min(img_h, rmax + pad)
        logger.info(f"   Mask bbox: ({cmin},{rmin})->({cmax},{rmax}), crop: ({x1},{y1})->({x2},{y2})")

        crop_img  = input_image.crop((x1, y1, x2, y2))
        crop_mask = mask_image.crop((x1, y1, x2, y2))
        crop_size = crop_img.size
        crop_img_512  = crop_img.resize((512, 512), Image.LANCZOS)
        crop_mask_512 = crop_mask.resize((512, 512), Image.NEAREST)
        mask_np = np.array(crop_mask_512)
        kernel  = np.ones((9, 9), np.uint8)
        mask_np = cv2.dilate(mask_np, kernel, iterations=1)
        crop_mask_512 = Image.fromarray(mask_np)

        inpainted_crop = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=crop_img_512,
            mask_image=crop_mask_512,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            width=512,
            height=512,
        ).images[0]

        inpainted_crop = inpainted_crop.resize(crop_size, Image.LANCZOS)
        output = input_image.copy()
        paste_mask = crop_mask.resize(crop_size, Image.NEAREST).convert("L")
        output.paste(inpainted_crop, (x1, y1), paste_mask)
        logger.info("   OK Inpainting completed (crop-inpaint-paste)!")

        saved_path = self._save_result(
            original_image=input_image, result_image=output,
            mask_image=mask_image, prompt=prompt, scenario=scenario, seed=seed,
        )
        logger.info(f"   OK Result saved: {saved_path}")
        return output, mask_image, seed

    def _save_result(self, original_image, result_image, mask_image, prompt, scenario, seed):
        """
        Save result PNG and side-by-side composite JPG to SAVE_DIR.
        Files: <ts>_<scenario>_seed<seed>_result.png
               <ts>_<scenario>_seed<seed>_composite.jpg  [orig|mask|result]
        """
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{ts}_{scenario}_seed{seed}"
        result_path = SAVE_DIR / f"{base}_result.png"
        result_image.save(str(result_path))
        try:
            w, h = result_image.size
            orig_r = original_image.resize((w, h), Image.LANCZOS)
            mask_r = mask_image.convert("RGB").resize((w, h), Image.NEAREST)
            composite = Image.new("RGB", (w * 3, h))
            composite.paste(orig_r,       (0,     0))
            composite.paste(mask_r,       (w,     0))
            composite.paste(result_image, (w * 2, 0))
            composite.save(str(SAVE_DIR / f"{base}_composite.jpg"), quality=95)
        except Exception as e:
            logger.warning(f"   WARNING Could not save composite: {e}")
        return str(result_path)


editor = BatikInpaintingEditor()


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "Batik Inpainting API",
        "method": "Inpainting (Precise Editing)",
        "base_model": editor.base_model,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_scenario": editor.current_scenario,
        "eval_available": _EVAL_AVAILABLE,
        "save_dir": str(SAVE_DIR),
    }), 200


@app.route('/inpaint', methods=['POST'])
def inpaint_motif():
    """
    Required: image (file), mask (file), prompt (str), scenario (str)
    Optional: steps, guidance_scale, strength, seed, negative_prompt,
              return_mask (bool), evaluate (bool)
    Returns: JPEG image + X-Seed / X-Saved-Path / X-Eval-* headers
    """
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        if 'mask' not in request.files:
            return jsonify({"error": "No mask file provided"}), 400

        image_file = request.files['image']
        mask_file  = request.files['mask']
        if image_file.filename == '' or mask_file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        prompt   = request.form.get('prompt')
        scenario = request.form.get('scenario')
        if not prompt:
            return jsonify({"error": "Missing required field: prompt"}), 400
        if not scenario:
            return jsonify({"error": "Missing required field: scenario"}), 400
        if scenario not in editor.lora_paths:
            return jsonify({"error": f"Invalid scenario. Must be one of: {list(editor.lora_paths.keys())}"}), 400

        try:
            steps          = int(request.form.get('steps', 30))
            guidance_scale = float(request.form.get('guidance_scale', 7.5))
            strength       = float(request.form.get('strength', 0.99))
            seed           = int(request.form.get('seed', -1))
            negative_prompt = request.form.get('negative_prompt', 'blurry, bad quality, distorted')
            return_mask     = request.form.get('return_mask', 'false').lower() == 'true'
            run_eval        = request.form.get('evaluate', 'false').lower() == 'true'
        except ValueError as e:
            return jsonify({"error": f"Invalid parameter type: {str(e)}"}), 400

        if not 0.0 <= strength <= 1.0:
            return jsonify({"error": "strength must be between 0.0 and 1.0"}), 400

        input_image = Image.open(image_file.stream).convert('RGB')
        mask_image  = Image.open(mask_file.stream).convert('L')
        logger.info(f"Received image: {input_image.size}, mask: {mask_image.size}")

        # ── simpan referensi mask ASLI sebelum diproses pipeline ──
        original_mask_for_eval = mask_image.copy()

        try:
            inpainted_image, mask_used, actual_seed = editor.inpaint_image(
                input_image=input_image, mask_image=mask_image,
                prompt=prompt, scenario=scenario,
                negative_prompt=negative_prompt, steps=steps,
                guidance_scale=guidance_scale, strength=strength, seed=seed,
            )

            if return_mask:
                buf = BytesIO()
                mask_used.save(buf, format='PNG')
                buf.seek(0)
                return send_file(buf, mimetype='image/png', as_attachment=False, download_name='mask.png')

            buf = BytesIO()
            inpainted_image.save(buf, format='JPEG', quality=95)
            buf.seek(0)

            eval_scores = {}
            if run_eval and _EVAL_AVAILABLE:
                logger.info("   Running evaluation metrics...")
                # Pastikan semua image sama size untuk evaluasi
                eval_size = input_image.size
                inpainted_eval = inpainted_image.resize(eval_size, Image.LANCZOS)
                mask_eval = original_mask_for_eval.resize(eval_size, Image.NEAREST).convert("RGB")

                # Debug: cek apakah mask benar-benar ada isinya
                mask_np_check = np.array(original_mask_for_eval)
                white_pixels = np.sum(mask_np_check > 127)
                total_pixels = mask_np_check.size
                logger.info(f"   Mask check: {white_pixels}/{total_pixels} white pixels ({100*white_pixels/total_pixels:.1f}%)")

                for metric, fn, args in [
                    ("clip_score",     compute_clip_score,     (inpainted_eval, prompt)),
                    ("nima_score",     compute_nima_score,     (inpainted_eval,)),
                    ("ssim_non_mask",  compute_ssim_non_mask,  (input_image, inpainted_eval, mask_eval)),
                    ("lpips_non_mask", compute_lpips_non_mask, (input_image, inpainted_eval, mask_eval)),
                ]:
                    try:
                        eval_scores[metric] = fn(*args)
                        logger.info(f"   {metric:20s}: {eval_scores[metric]}")
                    except Exception as e:
                        eval_scores[metric] = None
                        logger.warning(f"   {metric} error: {e}")
                ts2 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                record = {"timestamp": ts2, "scenario": scenario, "seed": actual_seed,
                          "prompt": prompt, "steps": steps, "guidance_scale": guidance_scale, **eval_scores}
                json_path = SAVE_DIR / f"{ts2}_{scenario}_seed{actual_seed}_eval.json"
                with open(str(json_path), "w", encoding="utf-8") as f:
                    json.dump(record, f, indent=2, ensure_ascii=False)
                logger.info(f"   OK Eval JSON saved: {json_path}")
            elif run_eval and not _EVAL_AVAILABLE:
                logger.warning("   evaluate=true requested but utils/evaluation.py unavailable")

            response = send_file(buf, mimetype='image/jpeg', as_attachment=False, download_name='inpainted.jpg')
            response.headers['X-Seed'] = str(actual_seed)
            response.headers['X-Saved-Path'] = str(SAVE_DIR)
            response.headers['Cache-Control'] = 'no-cache'
            for k, v in eval_scores.items():
                if v is not None:
                    response.headers[f'X-Eval-{k.replace("_", "-")}'] = str(v)
            return response

        except Exception as e:
            logger.error(f"Error during inpainting: {e}")
            logger.error(traceback.format_exc())
            return jsonify({"error": f"Inpainting failed: {str(e)}"}), 500

    except Exception as e:
        logger.error(f"Request processing error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/evaluate', methods=['POST'])
def evaluate_existing():
    """
    Evaluate already-generated images independently.
    Required: original (file), result (file), mask (file), prompt (str)
    Optional: scenario (str), save (bool, default true)
    Returns: JSON with clip_score, nima_score, ssim_non_mask, lpips_non_mask
    """
    if not _EVAL_AVAILABLE:
        return jsonify({"error": "Evaluation module not available. Ensure utils/evaluation.py exists."}), 503
    try:
        for field in ('original', 'result', 'mask'):
            if field not in request.files:
                return jsonify({"error": f"Missing required file field: '{field}'"}), 400
        if not request.form.get('prompt'):
            return jsonify({"error": "Missing required field: 'prompt'"}), 400

        original_img = Image.open(request.files['original'].stream).convert('RGB')
        result_img   = Image.open(request.files['result'].stream).convert('RGB')
        mask_img     = Image.open(request.files['mask'].stream).convert('RGB')
        prompt_val   = request.form.get('prompt')
        scenario_val = request.form.get('scenario', 'unknown')
        do_save      = request.form.get('save', 'true').lower() == 'true'

        logger.info(f"Standalone evaluation | scenario={scenario_val}")
        scores = {}
        for metric, fn, args in [
            ("clip_score",     compute_clip_score,     (result_img, prompt_val)),
            ("nima_score",     compute_nima_score,     (result_img,)),
            ("ssim_non_mask",  compute_ssim_non_mask,  (original_img, result_img, mask_img)),
            ("lpips_non_mask", compute_lpips_non_mask, (original_img, result_img, mask_img)),
        ]:
            try:
                scores[metric] = fn(*args)
                logger.info(f"  {metric:20s}: {scores[metric]}")
            except Exception as e:
                scores[metric] = None
                logger.warning(f"  {metric} error: {e}")

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        body = {"timestamp": ts, "scenario": scenario_val, "prompt": prompt_val, **scores}
        if do_save:
            SAVE_DIR.mkdir(parents=True, exist_ok=True)
            json_path = SAVE_DIR / f"{ts}_{scenario_val}_eval.json"
            with open(str(json_path), "w", encoding="utf-8") as f:
                json.dump(body, f, indent=2, ensure_ascii=False)
            body["saved_to"] = str(json_path)
            logger.info(f"  OK Eval JSON saved: {json_path}")
        return jsonify(body), 200

    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "service": "Batik Inpainting API",
        "version": "1.1",
        "description": "Precise motif editing with inpainting",
        "endpoints": {
            "inpaint":  "/inpaint  [POST] - Inpaint (auto-save + optional eval)",
            "evaluate": "/evaluate [POST] - Evaluate existing images (CLIP/NIMA/SSIM/LPIPS)",
            "health":   "/health   [GET]  - Health check",
        },
        "available_scenarios": list(editor.lora_paths.keys()),
        "usage": {
            "method": "POST", "endpoint": "/inpaint",
            "content_type": "multipart/form-data",
            "required_fields": ["image", "mask", "prompt", "scenario"],
            "optional_fields": {
                "steps": "integer (default: 30)",
                "guidance_scale": "float (default: 7.5)",
                "strength": "0.0-1.0 (default: 0.99)",
                "seed": "integer (default: -1 for random)",
                "negative_prompt": "string",
                "return_mask": "boolean (default: false)",
                "evaluate": "boolean — run CLIP/NIMA/SSIM/LPIPS after inpainting",
            }
        },
        "mask_format": {"white_pixels": "Area to inpaint", "black_pixels": "Area to keep"},
        "save_dir": str(SAVE_DIR),
        "eval_available": _EVAL_AVAILABLE,
    }), 200


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Starting Batik Inpainting API Server")
    logger.info("=" * 60)
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            logger.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    logger.info(f"Available scenarios: {list(editor.lora_paths.keys())}")
    logger.info("API Endpoints:")
    logger.info("  POST /inpaint   - Inpaint (auto-save + optional eval)")
    logger.info("  POST /evaluate  - Evaluate existing images")
    logger.info("  GET  /health    - Health check")
    logger.info("  GET  /          - API info")
    logger.info(f"Results will be saved to: {SAVE_DIR}")
    logger.info(f"Evaluation available   : {_EVAL_AVAILABLE}")
    logger.info("=" * 60)
    app.run(host='0.0.0.0', port=8007, debug=False)
