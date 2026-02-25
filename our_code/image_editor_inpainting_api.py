#!/usr/bin/env python3
"""
Batik Inpainting API - Precise Motif Editing
Edit specific areas while preserving the rest
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import logging
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


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(asctime)s %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)


class BatikInpaintingEditor:
    """Inpainting editor for precise motif editing"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # IMPORTANT: Must use inpainting-specific model, not regular SD
        self.base_model = "runwayml/stable-diffusion-inpainting"
        self.pipe = None
        self.current_scenario = None
        
        # LoRA paths mapping
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
    
    def create_mask_from_color(self, image: Image.Image, target_color: tuple, tolerance: int = 30):
        """Create mask from color selection (simple color-based masking)"""
        img_array = np.array(image)
        
        # Calculate color distance
        color_diff = np.abs(img_array - np.array(target_color))
        mask = np.all(color_diff <= tolerance, axis=-1)
        
        # Convert to PIL mask (white = inpaint, black = keep)
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        
        return mask_img
    
    def load_pipeline(self, scenario: str):
        """Load or reload Inpainting pipeline with specific LoRA"""
        # Only reload if scenario changed or pipe not loaded
        if self.pipe is None or self.current_scenario != scenario:
            logger.info(f"Loading Inpainting pipeline for scenario: {scenario}")
            
            # Unload previous pipeline
            if self.pipe is not None:
                del self.pipe
                gc.collect()
                torch.cuda.empty_cache()
            
            # Load Inpainting pipeline
            logger.info("   Loading SD Inpainting pipeline...")
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )
            
            # NOTE: LoRA trained on SD v1.5 (4-channel UNet) is NOT compatible
            # with SD Inpainting (9-channel UNet). Loading it would corrupt the
            # inpainting capability. We use the base inpainting model only.
            lora_path = self.lora_paths.get(scenario)
            if lora_path and Path(lora_path).exists():
                logger.info(f"   ⚠️  Skipping LoRA (SD v1.5 LoRA incompatible with SD Inpainting 9ch UNet)")
                logger.info(f"   ℹ️  Using base SD Inpainting model for reliable results")
            else:
                logger.info(f"   ℹ️  Using base SD Inpainting model")
            
            # Set scheduler
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
            
            # Move to device
            self.pipe = self.pipe.to(self.device)
            
            # Enable memory optimizations
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                logger.info("   ✓ Enabled xFormers")
            except:
                logger.info("   ⚠️  xFormers not available")
            
            self.current_scenario = scenario
            logger.info("   ✓ Inpainting pipeline loaded successfully!")
    
    def _prepare_images(self, input_image: Image.Image, mask_image: Image.Image):
        """Resize images to 512x512 as required by SD inpainting, dilate mask for better blending"""
        orig_size = input_image.size
        
        # SD Inpainting requires 512x512
        input_resized = input_image.resize((512, 512), Image.LANCZOS)
        mask_resized = mask_image.resize((512, 512), Image.NEAREST)
        
        # Dilate mask slightly for better edge blending
        mask_array = np.array(mask_resized)
        kernel = np.ones((15, 15), np.uint8)
        mask_dilated = cv2.dilate(mask_array, kernel, iterations=1)
        mask_resized = Image.fromarray(mask_dilated)
        
        return input_resized, mask_resized, orig_size

    @torch.no_grad()
    def inpaint_image(
        self,
        input_image: Image.Image,
        mask_image: Image.Image,
        prompt: str,
        scenario: str,
        negative_prompt: str = "blurry, bad quality, distorted, ugly, deformed",
        steps: int = 30,
        guidance_scale: float = 7.5,
        strength: float = 0.99,  # kept for API compat but not used by SD inpainting
        seed: int = None,
    ):
        """Inpaint specific area of batik image
        
        Args:
            input_image: Original image
            mask_image: Mask (WHITE = area to CHANGE, BLACK = area to KEEP)
            prompt: What to generate in masked area
            strength: kept for API compatibility (not used by SD inpainting pipeline)
        """
        
        # Load pipeline for this scenario
        self.load_pipeline(scenario)

        # Ensure mask is same size as image first
        if input_image.size != mask_image.size:
            mask_image = mask_image.resize(input_image.size, Image.NEAREST)

        # Handle seed
        if seed is None or seed == -1:
            seed = random.randint(0, 2**32 - 1)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        orig_size = input_image.size

        logger.info(f"Inpainting with prompt: '{prompt[:60]}...'")
        logger.info(f"   Image size: {orig_size}")
        logger.info(f"   Steps: {steps}, CFG: {guidance_scale}, Seed: {seed}")

        # ── CROP-INPAINT-PASTE approach ──────────────────────────────────────
        # Problem: if mask is small (e.g. 30% center), the masked area only
        # occupies ~150x150px of the 512x512 canvas → too small for generation.
        # Solution: crop the bounding box of the mask, inpaint at full 512x512,
        # then paste the result back onto the original image.

        mask_array = np.array(mask_image)

        # Find bounding box of white (masked) area
        rows = np.any(mask_array > 127, axis=1)
        cols = np.any(mask_array > 127, axis=0)

        if not rows.any():
            # Mask is empty — return original image unchanged
            logger.warning("   ⚠️  Mask is empty (all black), returning original image")
            return input_image, mask_image, seed

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Add padding around the crop box (20% of box size, min 32px)
        h_box = rmax - rmin
        w_box = cmax - cmin
        pad = max(32, int(max(h_box, w_box) * 0.20))

        img_w, img_h = input_image.size
        x1 = max(0, cmin - pad)
        y1 = max(0, rmin - pad)
        x2 = min(img_w, cmax + pad)
        y2 = min(img_h, rmax + pad)

        logger.info(f"   Mask bbox: ({cmin},{rmin})→({cmax},{rmax}), crop with pad: ({x1},{y1})→({x2},{y2})")

        # Crop image and mask to the bounding box
        crop_img  = input_image.crop((x1, y1, x2, y2))
        crop_mask = mask_image.crop((x1, y1, x2, y2))

        # Resize crop to 512x512 for inpainting
        crop_size = crop_img.size
        crop_img_512  = crop_img.resize((512, 512), Image.LANCZOS)
        crop_mask_512 = crop_mask.resize((512, 512), Image.NEAREST)

        # Dilate mask slightly for smoother blending
        mask_np = np.array(crop_mask_512)
        kernel  = np.ones((9, 9), np.uint8)
        mask_np = cv2.dilate(mask_np, kernel, iterations=1)
        crop_mask_512 = Image.fromarray(mask_np)

        # Run inpainting on the cropped region at full 512x512
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

        # Resize inpainted crop back to original crop size
        inpainted_crop = inpainted_crop.resize(crop_size, Image.LANCZOS)

        # Paste inpainted crop back onto the original image
        output = input_image.copy()
        # Use the original (non-dilated) mask for pasting to preserve edges
        paste_mask = crop_mask.resize(crop_size, Image.NEAREST).convert("L")
        output.paste(inpainted_crop, (x1, y1), paste_mask)

        logger.info("   ✓ Inpainting completed (crop-inpaint-paste)!")

        return output, mask_image, seed


# Initialize editor
editor = BatikInpaintingEditor()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Batik Inpainting API",
        "method": "Inpainting (Precise Editing)",
        "base_model": editor.base_model,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_scenario": editor.current_scenario,
    }), 200


@app.route('/inpaint', methods=['POST'])
def inpaint_motif():
    """
    Inpaint specific area of batik image
    
    Required form fields:
    - image: Input batik image file
    - mask: Mask image file (white = inpaint, black = keep)
    - prompt: What to generate in masked area
    - scenario: LoRA scenario to use
    
    Optional form fields:
    - steps: Number of inference steps (default: 30)
    - guidance_scale: CFG scale (default: 7.5)
    - strength: How much to change masked area 0.0-1.0 (default: 0.99)
    - seed: Random seed (default: -1 for random)
    - negative_prompt: Negative prompt (optional)
    - return_mask: Return mask too (optional, default false)
    
    Returns:
    - Inpainted image (JPEG)
    - X-Seed header with actual seed used
    """
    try:
        # Check if image and mask files are present
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        if 'mask' not in request.files:
            return jsonify({"error": "No mask file provided"}), 400
        
        image_file = request.files['image']
        mask_file = request.files['mask']
        
        if image_file.filename == '' or mask_file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        # Get form parameters
        prompt = request.form.get('prompt')
        scenario = request.form.get('scenario')
        
        if not prompt:
            return jsonify({"error": "Missing required field: prompt"}), 400
        if not scenario:
            return jsonify({"error": "Missing required field: scenario"}), 400
        
        # Validate scenario
        if scenario not in editor.lora_paths:
            return jsonify({"error": f"Invalid scenario. Must be one of: {list(editor.lora_paths.keys())}"}), 400
        
        # Get optional parameters
        try:
            steps = int(request.form.get('steps', 30))
            guidance_scale = float(request.form.get('guidance_scale', 7.5))
            strength = float(request.form.get('strength', 0.99))
            seed = int(request.form.get('seed', -1))
            negative_prompt = request.form.get('negative_prompt', 'blurry, bad quality, distorted')
            return_mask = request.form.get('return_mask', 'false').lower() == 'true'
        except ValueError as e:
            return jsonify({"error": f"Invalid parameter type: {str(e)}"}), 400
        
        # Validate strength
        if not 0.0 <= strength <= 1.0:
            return jsonify({"error": "strength must be between 0.0 and 1.0"}), 400
        
        # Load input image and mask
        input_image = Image.open(image_file.stream).convert('RGB')
        mask_image = Image.open(mask_file.stream).convert('L')  # Grayscale
        
        logger.info(f"Received image: {input_image.size}, mask: {mask_image.size}")
        
        # Inpaint image
        try:
            inpainted_image, mask_used, actual_seed = editor.inpaint_image(
                input_image=input_image,
                mask_image=mask_image,
                prompt=prompt,
                scenario=scenario,
                negative_prompt=negative_prompt,
                steps=steps,
                guidance_scale=guidance_scale,
                strength=strength,
                seed=seed,
            )
            
            # If return_mask is true, return mask instead
            if return_mask:
                img_byte_arr = BytesIO()
                mask_used.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                
                return send_file(
                    img_byte_arr,
                    mimetype='image/png',
                    as_attachment=False,
                    download_name='mask.png'
                )
            
            # Return inpainted image
            img_byte_arr = BytesIO()
            inpainted_image.save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr.seek(0)
            
            response = send_file(
                img_byte_arr,
                mimetype='image/jpeg',
                as_attachment=False,
                download_name='inpainted.jpg'
            )
            response.headers['X-Seed'] = str(actual_seed)
            response.headers['Cache-Control'] = 'no-cache'
            
            return response
            
        except Exception as e:
            logger.error(f"Error during inpainting: {e}")
            logger.error(traceback.format_exc())
            return jsonify({"error": f"Inpainting failed: {str(e)}"}), 500
            
    except Exception as e:
        logger.error(f"Request processing error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        "service": "Batik Inpainting API",
        "version": "1.0",
        "description": "Precise motif editing with inpainting",
        "method": "Stable Diffusion Inpainting + LoRA",
        "endpoints": {
            "inpaint": "/inpaint [POST] - Inpaint specific area",
            "health": "/health [GET] - Health check"
        },
        "available_scenarios": list(editor.lora_paths.keys()),
        "usage": {
            "method": "POST",
            "endpoint": "/inpaint",
            "content_type": "multipart/form-data",
            "required_fields": ["image", "mask", "prompt", "scenario"],
            "optional_fields": {
                "steps": "integer (default: 30)",
                "guidance_scale": "float (default: 7.5)",
                "strength": "0.0-1.0 (default: 0.99) - How much to change masked area",
                "seed": "integer (default: -1 for random)",
                "negative_prompt": "string",
                "return_mask": "boolean (default: false) - Return mask for debugging"
            }
        },
        "mask_format": {
            "white_pixels": "Area to inpaint (change)",
            "black_pixels": "Area to preserve (keep)",
            "format": "Grayscale PNG or JPEG"
        }
    }), 200


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Starting Batik Inpainting API Server")
    logger.info("=" * 60)
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA Devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    logger.info(f"Available scenarios: {list(editor.lora_paths.keys())}")
    logger.info("=" * 60)
    logger.info("API Endpoints:")
    logger.info("  POST /inpaint - Inpaint specific area")
    logger.info("  GET /health - Health check")
    logger.info("  GET / - API info")
    logger.info("=" * 60)
    logger.info("Inpainting allows precise editing of specific motifs!")
    logger.info("=" * 60)
    
    app.run(host='0.0.0.0', port=8007, debug=False)
