"""
Evaluation module for PILOT inpainting pipeline.

Metrics:
  - CLIP Score   : Semantic alignment between generated image and text prompt
  - NIMA Score   : Neural aesthetic / quality score of the generated image
  - SSIM         : Structural similarity on the NON-masked (background) area only
  - LPIPS        : Perceptual similarity on the NON-masked (background) area only

Usage (standalone):
    from utils.evaluation import evaluate_results
    metrics = evaluate_results(image_list, original_image, mask_image, prompt, output_path)
"""

import os
import json
import datetime
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# Helper: lazy-load heavy libraries so the rest of the pipeline is unaffected
# ──────────────────────────────────────────────────────────────────────────────

def _load_clip():
    """Return (model, preprocess, tokenize, device) for open_clip."""
    try:
        import open_clip
    except ImportError:
        raise ImportError(
            "open_clip is not installed. Run: pip install open-clip-torch"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model = model.to(device).eval()
    tokenize = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess, tokenize, device


def _load_lpips():
    """Return lpips loss network."""
    try:
        import lpips
    except ImportError:
        raise ImportError("lpips is not installed. Run: pip install lpips")
    loss_fn = lpips.LPIPS(net="alex")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = loss_fn.to(device)
    return loss_fn, device


def _load_nima():
    """
    Return a lightweight NIMA scorer using torchvision MobileNetV2.
    Weights are from the unofficial NIMA checkpoint hosted on GitHub.
    Falls back to None if download fails so the rest of evaluation still runs.
    """
    try:
        import torchvision.models as models
        import torch.nn.functional as F
    except ImportError:
        raise ImportError("torchvision is required for NIMA.")

    class NIMAModel(nn.Module):
        def __init__(self):
            super().__init__()
            base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            self.features = base.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Dropout(0.75),
                nn.Linear(base.last_channel, 10),
                nn.Softmax(dim=1),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NIMAModel().to(device).eval()

    # Try to load pretrained NIMA weights
    weight_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "nima_mobilenet.pth"
    )
    weight_path = os.path.normpath(weight_path)

    if os.path.exists(weight_path):
        state = torch.load(weight_path, map_location=device)
        # Some checkpoints wrap weights under a key
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        try:
            model.load_state_dict(state, strict=False)
            print("[NIMA] Loaded pretrained weights from", weight_path)
        except Exception as e:
            print(f"[NIMA] Warning: could not load weights ({e}). Using ImageNet init.")
    else:
        print(
            f"[NIMA] Warning: pretrained weights not found at '{weight_path}'.\n"
            "       Download from https://github.com/truskovskiyk/nima.pytorch "
            "and save as models/nima_mobilenet.pth for best results.\n"
            "       Proceeding with ImageNet-initialised backbone (scores may not be meaningful)."
        )

    return model, device


# ──────────────────────────────────────────────────────────────────────────────
# Metric implementations
# ──────────────────────────────────────────────────────────────────────────────

def compute_clip_score(generated_image: Image.Image, prompt: str) -> float:
    """
    Compute CLIP Score between a generated PIL image and a text prompt.

    Score = cosine_similarity(image_features, text_features) * 100
    Higher is better (range roughly 0-100).

    Args:
        generated_image: PIL.Image of the inpainted result.
        prompt: The text prompt used for inpainting.

    Returns:
        float: CLIP score value.
    """
    model, preprocess, tokenize, device = _load_clip()

    img_tensor = preprocess(generated_image).unsqueeze(0).to(device)
    text_tokens = tokenize([prompt]).to(device)

    with torch.no_grad():
        img_features = model.encode_image(img_tensor)
        txt_features = model.encode_text(text_tokens)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)
        score = (img_features * txt_features).sum(dim=-1).item()

    return round(float(score), 4)


def compute_nima_score(generated_image: Image.Image) -> float:
    """
    Compute NIMA (Neural Image Assessment) aesthetic score.

    Uses a MobileNetV2 backbone that predicts a distribution over 10 quality
    ratings (1-10). The final score is the mean of the distribution (MOS).
    Higher is better (range 1-10).

    Args:
        generated_image: PIL.Image of the inpainted result.

    Returns:
        float: NIMA mean opinion score (1-10).
    """
    import torchvision.transforms as T

    model, device = _load_nima()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = transform(generated_image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        dist = model(img_tensor)  # shape: (1, 10)

    # Mean opinion score: sum(rating * probability) for rating in 1..10
    ratings = torch.arange(1, 11, dtype=torch.float32).to(device)
    mos = (dist * ratings).sum(dim=1).item()
    return round(float(mos), 4)


def compute_ssim_non_mask(
    original_image: Image.Image,
    generated_image: Image.Image,
    mask_image: Image.Image,
) -> float:
    """
    Compute SSIM only on the NON-masked (background) area.

    A high score means the inpainting preserved the background well.
    Range: -1 to 1 (higher is better).

    Args:
        original_image : PIL.Image — the original image before inpainting.
        generated_image: PIL.Image — the inpainted result.
        mask_image     : PIL.Image — white = inpainted region, black = background.

    Returns:
        float: SSIM score on background pixels.
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        raise ImportError(
            "scikit-image is not installed. Run: pip install scikit-image"
        )

    orig_np = np.array(original_image.convert("RGB")).astype(np.float32) / 255.0
    gen_np = np.array(generated_image.convert("RGB")).astype(np.float32) / 255.0
    mask_np = np.array(mask_image.convert("L")).astype(np.float32) / 255.0

    # Non-mask: pixels where mask == 0 (black = background)
    bg_mask = (mask_np < 0.5).astype(np.float32)  # shape: H x W

    # Apply mask to both images (set masked region to 0 so it's excluded)
    orig_masked = orig_np * bg_mask[:, :, np.newaxis]
    gen_masked = gen_np * bg_mask[:, :, np.newaxis]

    # Compute SSIM per channel then average
    score_per_channel = []
    for c in range(3):
        s = ssim(
            orig_masked[:, :, c],
            gen_masked[:, :, c],
            data_range=1.0,
        )
        score_per_channel.append(s)

    return round(float(np.mean(score_per_channel)), 4)


def compute_lpips_non_mask(
    original_image: Image.Image,
    generated_image: Image.Image,
    mask_image: Image.Image,
) -> float:
    """
    Compute LPIPS perceptual distance only on the NON-masked (background) area.

    Lower is better (range 0-1).

    Args:
        original_image : PIL.Image — the original image before inpainting.
        generated_image: PIL.Image — the inpainted result.
        mask_image     : PIL.Image — white = inpainted region, black = background.

    Returns:
        float: LPIPS distance on background pixels.
    """
    loss_fn, device = _load_lpips()

    def pil_to_tensor(img: Image.Image) -> torch.Tensor:
        """Convert PIL Image to [-1, 1] float tensor (1, C, H, W)."""
        arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return (tensor * 2 - 1).to(device)

    orig_t = pil_to_tensor(original_image)
    gen_t = pil_to_tensor(generated_image)

    # Build background mask tensor: 1 = background, 0 = inpainted region
    mask_np = np.array(mask_image.convert("L")).astype(np.float32) / 255.0
    bg_mask = (mask_np < 0.5).astype(np.float32)
    bg_mask_t = torch.from_numpy(bg_mask).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
    bg_mask_t = bg_mask_t.expand_as(orig_t)

    # Zero-out the inpainted region so LPIPS focuses on background
    orig_bg = orig_t * bg_mask_t
    gen_bg = gen_t * bg_mask_t

    with torch.no_grad():
        dist = loss_fn(orig_bg, gen_bg).item()

    return round(float(dist), 4)


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation function (called from run_example.py)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_results(
    image_list: list,
    original_image: Image.Image,
    mask_image: Image.Image,
    prompt: str,
    output_path: str,
    config_name: str = "unknown",
) -> list:
    """
    Run all evaluation metrics on the generated image list and save results.

    This function is designed to be called right after the PILOT pipeline
    produces `image_list`. It does NOT modify any pipeline code.

    Args:
        image_list    : List[PIL.Image] — output from pipe().
        original_image: PIL.Image — original input image.
        mask_image    : PIL.Image — binary mask (white=inpaint, black=background).
        prompt        : str — text prompt used for generation.
        output_path   : str — directory to save evaluation_results.json.
        config_name   : str — name tag for the experiment (e.g. "t2i_step100").

    Returns:
        List[dict] — one dict of scores per generated image.
    """
    print("\n" + "=" * 60)
    print("  PILOT Evaluation — Running Metrics")
    print("=" * 60)

    all_results = []

    for idx, gen_img in enumerate(image_list):
        print(f"\n[Image {idx + 1}/{len(image_list)}]")
        result = {
            "image_index": idx,
            "config": config_name,
            "prompt": prompt,
        }

        # ── CLIP Score ─────────────────────────────────────────────────────
        try:
            clip_score = compute_clip_score(gen_img, prompt)
            result["clip_score"] = clip_score
            print(f"  CLIP Score         : {clip_score:.4f}  (higher is better, ~0-100)")
        except Exception as e:
            result["clip_score"] = None
            print(f"  CLIP Score         : ERROR — {e}")

        # ── NIMA Score ─────────────────────────────────────────────────────
        try:
            nima_score = compute_nima_score(gen_img)
            result["nima_score"] = nima_score
            print(f"  NIMA Score         : {nima_score:.4f}  (higher is better, 1-10)")
        except Exception as e:
            result["nima_score"] = None
            print(f"  NIMA Score         : ERROR — {e}")

        # ── SSIM (non-mask) ────────────────────────────────────────────────
        try:
            ssim_score = compute_ssim_non_mask(original_image, gen_img, mask_image)
            result["ssim_non_mask"] = ssim_score
            print(f"  SSIM (non-mask)    : {ssim_score:.4f}  (higher is better, -1 to 1)")
        except Exception as e:
            result["ssim_non_mask"] = None
            print(f"  SSIM (non-mask)    : ERROR — {e}")

        # ── LPIPS (non-mask) ───────────────────────────────────────────────
        try:
            lpips_score = compute_lpips_non_mask(original_image, gen_img, mask_image)
            result["lpips_non_mask"] = lpips_score
            print(f"  LPIPS (non-mask)   : {lpips_score:.4f}  (lower is better, 0-1)")
        except Exception as e:
            result["lpips_non_mask"] = None
            print(f"  LPIPS (non-mask)   : ERROR — {e}")

        all_results.append(result)

    # ── Save individual generated images ───────────────────────────────────
    for idx, gen_img in enumerate(image_list):
        gen_img_path = os.path.join(output_path, f"generated_{config_name}_{idx}.png")
        gen_img.save(gen_img_path)
        print(f"\n  Generated image saved: {gen_img_path}")

    # ── Save metrics to JSON ───────────────────────────────────────────────
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_path, f"eval_{config_name}_{timestamp}.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"  Evaluation complete. Results saved to:")
    print(f"  {json_path}")
    print("=" * 60 + "\n")

    return all_results
