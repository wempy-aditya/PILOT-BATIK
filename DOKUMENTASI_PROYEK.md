# Dokumentasi Proyek: PILOT — _Inpainting_ Motif Batik Berbasis Optimasi Ruang Laten

**Nama Proyek:** PILOT-BATIK  
**Repositori Asal:** [Lingzhi-Pan/PILOT](https://github.com/Lingzhi-Pan/PILOT)  
**Repositori Penelitian:** [wempy-aditya/PILOT-BATIK](https://github.com/wempy-aditya/PILOT-BATIK)  
**Tanggal Dokumentasi:** 25 Februari 2026

---

## Daftar Isi

1. [Latar Belakang & Tujuan Penelitian](#1-latar-belakang--tujuan-penelitian)
2. [Gambaran Umum Metode PILOT (Paper Asli)](#2-gambaran-umum-metode-pilot-paper-asli)
3. [Arsitektur Sistem Keseluruhan](#3-arsitektur-sistem-keseluruhan)
4. [Struktur Direktori Proyek](#4-struktur-direktori-proyek)
5. [Penjelasan Kode Sumber (Base Code Asli)](#5-penjelasan-kode-sumber-base-code-asli)
   - 5.1 [run_example.py](#51-run_examplepy--titik-masuk-utama)
   - 5.2 [pipeline/pipeline_pilot.py](#52-pipelinepipeline_pilotpy--inti-pipeline)
   - 5.3 [models/attn_processor.py](#53-modelsattn_processorpy--attention-processor)
   - 5.4 [utils/image_processor.py](#54-utilsimage_processorpy--pemrosesan-gambar)
   - 5.5 [utils/visualize.py](#55-utilsvisualizepy--visualisasi-hasil)
   - 5.6 [utils/generate_spatial_map.py](#56-utilsgenerate_spatial_mappy--kontrol-spasial)
   - 5.7 [configs/](#57-configs--konfigurasi-eksperimen)
6. [Alur Pipeline Lengkap (Diagram)](#6-alur-pipeline-lengkap-diagram)
7. [Penambahan Modul Evaluasi (Kontribusi Penelitian)](#7-penambahan-modul-evaluasi-kontribusi-penelitian)
   - 7.1 [utils/evaluation.py](#71-utilsevaluationpy--modul-evaluasi-baru)
   - 7.2 [Modifikasi run_example.py](#72-modifikasi-run_examplepy)
   - 7.3 [Update requirements.txt](#73-update-requirementstxt)
8. [Penjelasan Detail Setiap Metrik Evaluasi](#8-penjelasan-detail-setiap-metrik-evaluasi)
9. [Output yang Dihasilkan](#9-output-yang-dihasilkan)
10. [Cara Menjalankan Program](#10-cara-menjalankan-program)
11. [Konfigurasi Parameter](#11-konfigurasi-parameter)
12. [Dependensi dan Lingkungan](#12-dependensi-dan-lingkungan)
13. [Rencana Eksperimen & Perbandingan](#13-rencana-eksperimen--perbandingan)
14. [Referensi](#14-referensi)

---

## 1. Latar Belakang & Tujuan Penelitian

### 1.1 Motivasi

Motif batik merupakan warisan budaya Indonesia yang kaya akan pola dan ornamen khas. Pengeditan motif batik secara digital—misalnya menambah, mengubah, atau memperbaiki area tertentu dari kain batik—membutuhkan pendekatan yang mampu:

1. **Mempertahankan konsistensi visual** antara area yang diedit dan latar belakang asli.
2. **Mengikuti arahan teks** (_text-guided_) yang mendeskripsikan motif yang diinginkan.
3. **Menjaga kekhasan batik**, seperti pola geometris, repetisi, dan nuansa warna khas.

### 1.2 Pendekatan Solusi

Penelitian ini menggunakan **PILOT** (_Inpainting via Latent Space Optimization_) sebagai _baseline_, yaitu sebuah metode inpainting berbasis model difusi yang **tidak memerlukan fine-tuning** tambahan. PILOT mengoptimalkan secara langsung vektor laten pada ruang difusi untuk menghasilkan konten yang:

- Semantik sesuai dengan prompt teks yang diberikan.
- Koheren dengan area latar belakang yang tidak diedit.

### 1.3 Tujuan Penelitian

| No  | Tujuan                                                                          |
| --- | ------------------------------------------------------------------------------- |
| 1   | Menggunakan PILOT sebagai _baseline_ untuk tugas inpainting motif batik         |
| 2   | Mengevaluasi performa PILOT pada gambar batik menggunakan metrik kuantitatif    |
| 3   | Membandingkan hasil eksperimen bawaan PILOT dengan eksperimen pada gambar batik |
| 4   | Menambahkan modul evaluasi otomatis tanpa merombak kode dasar                   |

---

## 2. Gambaran Umum Metode PILOT (Paper Asli)

> **Judul Paper:** _Coherent and Multi-modality Image Inpainting via Latent Space Optimization_  
> **Penulis:** Lingzhi Pan, Tong Zhang, Bingyuan Chen, Qi Zhou, Wei Ke, Sabine Süsstrunk, Mathieu Salzmann  
> **Institusi:** EPFL & Xi'an Jiaotong University  
> **Publikasi:** arXiv:2407.08019 (2024)

### 2.1 Ide Inti

Metode PILOT berargumen bahwa **model difusi besar yang sudah ada sudah cukup kuat** untuk menghasilkan gambar realistis tanpa perlu fine-tuning tambahan. Alih-alih mengubah model, PILOT **mengoptimalkan vektor laten** $\mathbf{x}_t$ selama proses denoising menggunakan dua fungsi loss khusus:

$$\mathcal{L}_{total} = \mathcal{L}_{bg} + \lambda \cdot \mathcal{L}_{sc}$$

**1. Background Loss ($\mathcal{L}_{bg}$):**  
Memastikan area di luar mask tetap konsisten dengan gambar asli.

$$\mathcal{L}_{bg} = \left\| \hat{\mathbf{x}}_0^{(t)} \odot (1-M) - \mathbf{x}_{orig} \odot (1-M) \right\|^2$$

Di mana:

- $\hat{\mathbf{x}}_0^{(t)}$ = prediksi gambar bersih pada timestep $t$
- $M$ = binary mask (1 = area inpainting, 0 = background)
- $\mathbf{x}_{orig}$ = gambar asli ter-encode ke ruang laten

**2. Semantic Centralization Loss ($\mathcal{L}_{sc}$):**  
Mendorong cross-attention pada area mask agar terfokus pada token teks yang relevan, sehingga konten yang dihasilkan semantik sesuai dengan prompt.

### 2.2 Alur Optimasi pada Setiap Timestep

```
Untuk setiap timestep t dalam denoising loop:
  1. Jalankan UNet forward pass → prediksi noise
  2. Estimasi x̂₀ dari noise
  3. Hitung L_bg + L_sc
  4. Backprop → update latent xₜ
  5. Ulangi num_gradient_ops kali
  6. DDIM step → xₜ₋₁
```

### 2.3 Dukungan Multi-Modal

PILOT secara mulus terintegrasi dengan berbagai kondisi tambahan:

| Mode                       | Kondisi Tambahan         | Config File                   |
| -------------------------- | ------------------------ | ----------------------------- |
| Text-to-Image              | Hanya teks               | `t2i_step100.yaml`            |
| Text + Spatial             | ControlNet / T2I-Adapter | `controlnet_step100.yaml`     |
| Text + Referensi           | IP-Adapter               | `ipa_step100.yaml`            |
| Text + Spatial + Referensi | ControlNet + IP-Adapter  | `ipa_controlnet_step100.yaml` |

---

## 3. Arsitektur Sistem Keseluruhan

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SISTEM PILOT-BATIK                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  INPUT                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ Gambar Batik │  │  Mask Area   │  │     Prompt Teks          │  │
│  │  (PIL Image) │  │ (Binary PNG) │  │  "motif batik kawung..." │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────────┘  │
│         │                 │                      │                  │
│         └─────────────────┴──────────────────────┘                 │
│                           │                                         │
│                    ┌──────▼──────┐                                  │
│                    │ run_example │  ← Titik Masuk Utama             │
│                    │    .py      │                                  │
│                    └──────┬──────┘                                  │
│                           │                                         │
│         ┌─────────────────┼──────────────────────┐                 │
│         ▼                 ▼                      ▼                  │
│  ┌────────────┐   ┌──────────────┐   ┌─────────────────┐           │
│  │ControlNet  │   │  IP-Adapter  │   │  T2I-Adapter    │           │
│  │(opsional)  │   │ (opsional)   │   │  (opsional)     │           │
│  └────────────┘   └──────────────┘   └─────────────────┘           │
│         │                 │                      │                  │
│         └─────────────────┴──────────────────────┘                 │
│                           │                                         │
│                ┌──────────▼──────────┐                              │
│                │   PilotPipeline     │  ← Stable Diffusion v1.5    │
│                │  (DDIM + Optimasi   │                              │
│                │   Ruang Laten)      │                              │
│                └──────────┬──────────┘                              │
│                           │                                         │
│                    ┌──────▼──────┐                                  │
│                    │ image_list  │  ← List[PIL.Image]               │
│                    │(hasil PIL)  │                                  │
│                    └──────┬──────┘                                  │
│                           │                                         │
│         ┌─────────────────┴──────────────────────┐                 │
│         ▼                                         ▼                 │
│  ┌────────────────┐                    ┌───────────────────────┐    │
│  │  visualize.py  │                    │    evaluation.py      │    │
│  │ (Gabungkan     │                    │  ┌─────────────────┐  │    │
│  │  gambar untuk  │                    │  │  CLIP Score     │  │    │
│  │  laporan)      │                    │  │  NIMA Score     │  │    │
│  └────────┬───────┘                    │  │  SSIM non-mask  │  │    │
│           │                            │  │  LPIPS non-mask │  │    │
│           ▼                            │  └─────────────────┘  │    │
│  ┌────────────────┐                    └──────────┬────────────┘    │
│  │  seed*_step*.  │                               │                 │
│  │     png        │                    ┌──────────▼────────────┐    │
│  │  (visualisasi  │                    │  eval_*.json          │    │
│  │  komposit)     │                    │  generated_*.png      │    │
│  └────────────────┘                    └───────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Struktur Direktori Proyek

```
PILOT/
│
├── run_example.py              # Titik masuk utama (dimodifikasi: +evaluasi)
├── requirements.txt            # Dependensi Python (diperbarui: +4 library)
├── README.md                   # Dokumentasi singkat repositori asli
│
├── pipeline/
│   └── pipeline_pilot.py       # Core pipeline: PilotPipeline class (~1448 baris)
│
├── models/
│   ├── attn_processor.py       # Custom attention processor untuk IP-Adapter
│   └── nima_mobilenet.pth      # [Opsional] Pretrained NIMA weights
│
├── utils/
│   ├── evaluation.py           # [BARU] Modul evaluasi: CLIP, NIMA, SSIM, LPIPS
│   ├── image_processor.py      # Konversi PIL↔Tensor, masking gambar
│   ├── visualize.py            # Membuat gambar komposit untuk laporan
│   └── generate_spatial_map.py # Menghasilkan peta spasial untuk ControlNet
│
├── configs/
│   ├── t2i_step100.yaml        # Konfigurasi: Text-to-Image, 100 langkah
│   ├── t2i_step200.yaml        # Konfigurasi: Text-to-Image, 200 langkah
│   ├── controlnet_step100.yaml # Konfigurasi: ControlNet, 100 langkah
│   ├── ipa_step100.yaml        # Konfigurasi: IP-Adapter, 100 langkah
│   └── ipa_controlnet_step100.yaml  # Konfigurasi: IP-Adapter + ControlNet
│
├── data/
│   ├── Brendhi.jpg             # Gambar input contoh (bawaan repo)
│   └── batik_mask.png          # Mask contoh untuk area batik
│
├── assets/
│   └── *.png / *.jpg           # Gambar ilustrasi untuk README
│
└── outputs/                    # Direktori output (dibuat otomatis)
    ├── seed100_step100.png      # Visualisasi komposit (original+hasil)
    ├── generated_*.png          # [BARU] Gambar hasil murni (per eksperimen)
    └── eval_*.json              # [BARU] Hasil metrik evaluasi (per eksperimen)
```

> **Keterangan:**
>
> - `[BARU]` = file/folder yang ditambahkan dalam penelitian ini
> - `[Opsional]` = file tambahan yang perlu diunduh secara manual

---

## 5. Penjelasan Kode Sumber (Base Code Asli)

### 5.1 `run_example.py` — Titik Masuk Utama

File ini adalah skrip utama yang mengorkestrasi seluruh proses dari input hingga output.

**Alur eksekusi:**

```python
# 1. Parsing argumen & load konfigurasi YAML
config = OmegaConf.load(args.config_file)

# 2. Load dan preprocessing gambar input
image = Image.open(config.input_image).convert("RGB")
mask_image = Image.open(config.mask_image).convert("RGB")

# 3. [Opsional] Load model tambahan
controlnet = ControlNetModel.from_pretrained(...)   # jika ada
adapter = T2IAdapter.from_pretrained(...)           # jika ada

# 4. Load model dasar Stable Diffusion
pipe = PilotPipeline.from_pretrained(config.model_id, ...)

# 5. [Opsional] Load LoRA / IP-Adapter
pipe.load_lora_weights(...)
pipe.load_ip_adapter(...)

# 6. Jalankan pipeline
image_list = pipe(prompt=..., image=..., mask=..., ...)

# 7. [TAMBAHAN] Evaluasi otomatis
evaluate_results(image_list, image, mask_image, config.prompt, ...)

# 8. Visualisasi dan simpan hasil
new_image_list = t2i_visualize(image, mask_image, image_list)
new_image.save(file_path)
```

**Catatan penting:**  
Mask diproses secara manual pada baris 52–57 untuk memastikan hanya nilai murni hitam `(0,0,0)` atau putih `(255,255,255)`. Piksel dengan warna abu-abu atau warna lain diubah menjadi hitam (dianggap background).

---

### 5.2 `pipeline/pipeline_pilot.py` — Inti Pipeline

File terbesar (~1448 baris) yang berisi kelas `PilotPipeline`, turunan dari `StableDiffusionPipeline` milik Hugging Face Diffusers.

**Komponen utama:**

#### a. `prepare_mask_and_masked_image(image, mask)`

Mengonversi gambar dan mask PIL menjadi tensor PyTorch dengan nilai dalam rentang `[-1, 1]`, dan menerapkan mask ke gambar.

#### b. `encode_image(image, generator)`

Mengkodekan gambar input ke ruang laten menggunakan VAE (_Variational Autoencoder_):

$$\mathbf{z} = \mathcal{E}(x) \cdot 0.18215$$

Di mana faktor $0.18215$ adalah konstanta penskalaan standar VAE pada Stable Diffusion.

#### c. `optimize_xt(x, image, mask, t, ...)` — Fungsi Inti Kontribusi PILOT

Ini adalah fungsi inti yang membedakan PILOT dari pipeline difusi standar. Fungsi ini mengoptimalkan vektor laten $\mathbf{x}_t$ menggunakan gradient descent selama `num_gradient_ops` iterasi pada setiap timestep:

```
Untuk setiap iterasi optimasi:
  1. Forward pass UNet dengan x_t saat ini
  2. Prediksi x̂₀ menggunakan DDIM inversion formula
  3. Hitung L_bg = MSE(x̂₀ ⊙ (1-M), x_orig ⊙ (1-M))
  4. Hitung L_sc = -mean(cross_attention pada area mask)
  5. L_total = coef × L_bg + L_sc
  6. Backprop & update: x_t ← x_t - lr × ∇L_total
  7. Terapkan momentum untuk stabilisasi
```

#### d. Denoising Loop Utama (langkah 11 dalam pipeline)

```python
for i, t in enumerate(timesteps):
    no_op = True
    if (i % op_interval == 0):    # Lakukan optimasi setiap op_interval langkah
        no_op = False
    if (t < 1000 * (1 - gamma)):  # Hentikan optimasi di timestep akhir
        no_op = True
        # Re-inject noise dari gambar asli untuk konsistensi background
        noise_source_latents = scheduler.add_noise(image, noise, t)
        latents = latents * (mask <= 0.5) + noise_source_latents * (mask > 0.5)

    latents, noise_pred = self.optimize_xt(...)
    latents = self.scheduler.step(noise_pred, t, latents).prev_sample
```

#### e. Attention Mask Multi-Skala (langkah 10)

Untuk menjaga konsistensi semantik, dibuat attention mask pada 4 skala berbeda sesuai arsitektur UNet:

```python
for attn_size in [64, 32, 16, 8]:
    attention_mask[str(attn_size**2)] = F.interpolate(1 - mask, (attn_size, attn_size))
```

---

### 5.3 `models/attn_processor.py` — Attention Processor

Berisi fungsi `revise_pilot_unet_attention_forward()` yang mengganti mekanisme attention standar UNet dengan versi yang mendukung **IP-Adapter** dan **attention mask kustom**.

Modifikasi ini diperlukan agar:

1. IP-Adapter dapat menyuntikkan fitur visual dari gambar referensi.
2. Attention mask dapat mempengaruhi distribusi perhatian pada area mask.

---

### 5.4 `utils/image_processor.py` — Pemrosesan Gambar

Berisi tiga fungsi utilitas konversi tensor:

| Fungsi                    | Input              | Output             | Keterangan                   |
| ------------------------- | ------------------ | ------------------ | ---------------------------- |
| `preprocess_image(image)` | `PIL.Image`        | `Tensor [1,C,H,W]` | Normalisasi ke `[-1, 1]`     |
| `tensor2PIL(image)`       | `Tensor [1,C,H,W]` | `PIL.Image`        | De-normalisasi ke `[0, 255]` |
| `mask4image(image, mask)` | `Tensor, Tensor`   | `Tensor`           | Terapkan mask ke gambar      |

---

### 5.5 `utils/visualize.py` — Visualisasi Hasil

Membuat gambar komposit (horizontal) untuk keperluan laporan/perbandingan:

| Fungsi                    | Susunan Output                                          |
| ------------------------- | ------------------------------------------------------- |
| `t2i_visualize()`         | `[Original+Mask \| Hasil]` — 2 kolom × 512px            |
| `ipa_visualize()`         | `[Original+Mask \| Ref Image \| Hasil]` — 3 kolom       |
| `spatial_visualize()`     | `[Original+Mask \| Cond Map \| Hasil]` — 3 kolom        |
| `ipa_spatial_visualize()` | `[Original+Mask \| Ref \| Cond Map \| Hasil]` — 4 kolom |

Area mask pada gambar original divisualisasikan dengan **warna putih** menggunakan fungsi `whitemask4image()`.

---

### 5.6 `utils/generate_spatial_map.py` — Kontrol Spasial

Berisi fungsi `img2cond()` yang mengkonversi gambar menjadi peta kondisional untuk ControlNet. Mendukung berbagai mode seperti:

- **HED** (holistically-nested edge detection)
- **Lineart** (garis tepi artistik)
- **Openpose** (deteksi pose tubuh)
- **Segmentasi semantik** (via UperNet)

---

### 5.7 `configs/` — Konfigurasi Eksperimen

Setiap file YAML mendefinisikan satu skenario eksperimen. Contoh `t2i_step100.yaml`:

```yaml
model_path: "runwayml" # Direktori model lokal
output_path: "outputs" # Direktori output
model_id: "stable-diffusion-v1-5" # Model base yang digunakan
input_image: "data/Brendhi.jpg" # Gambar input
mask_image: "data/batik_mask.png" # Mask area yang akan diedit
prompt: "traditional batik ornament, golden mandala pattern..."
negative_prompt: "photorealistic, blurry"

# Parameter generasi
cfg: 7.5 # Classifier-free guidance scale
W: 512 # Lebar output (piksel)
H: 512 # Tinggi output (piksel)
num: 1 # Jumlah gambar yang dihasilkan
seed: 100 # Random seed untuk reproduksibilitas

# Parameter optimasi PILOT (inti metode)
step: 100 # Jumlah langkah denoising DDIM
op_interval: 10 # Interval langkah untuk optimasi
num_gradient_ops: 10 # Iterasi gradient per langkah
gamma: 1 # Threshold timestep untuk optimasi
lr: 0.025 # Learning rate optimasi laten
lr_warmup: 0.007 # Warmup learning rate
lr_f: "exp" # Jadwal learning rate (exp/linear)
coef: 150 # Koefisien background loss
coef_f: "linear" # Jadwal koefisien (linear/constant)
momentum: 0.7 # Momentum untuk gradient update
fp16: true # Gunakan presisi setengah (hemat VRAM)
```

---

## 6. Alur Pipeline Lengkap (Diagram)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ALUR PIPELINE PILOT                                  │
└─────────────────────────────────────────────────────────────────────────┘

TAHAP 1 — PERSIAPAN INPUT
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│ Input Image  │     │  Mask Image  │     │   Text Prompt    │
│ (PIL→Tensor) │     │ (PIL→Tensor) │     │ (CLIP Tokenizer) │
└──────┬───────┘     └──────┬───────┘     └────────┬─────────┘
       │                    │                      │
       ▼                    ▼                      ▼
  Encode via VAE      Resize & Binarize      CLIP Text Encoder
  z ∈ R^{4×64×64}    M ∈ {0,1}^{64×64}    e_text ∈ R^{77×768}


TAHAP 2 — INISIALISASI LATEN
┌─────────────────────────────────────────────────────┐
│  x_T ~ N(0, I)  ← Gaussian noise awal              │
│  Buat attention_mask[64²,32²,16²,8²] dari mask M   │
│  Hitung coef_scale berdasarkan luas area mask       │
└─────────────────────────────────────────────────────┘


TAHAP 3 — DENOISING LOOP (T=100 langkah, dari t=T ke t=0)
┌─────────────────────────────────────────────────────────────────┐
│  Untuk setiap timestep t:                                       │
│                                                                 │
│  ┌─── [Jika t >= 1000(1-γ) DAN i % op_interval == 0] ──────┐  │
│  │  OPTIMASI LATEN (num_gradient_ops kali):                  │  │
│  │                                                           │  │
│  │  Untuk setiap iterasi k:                                  │  │
│  │    1. UNet(x_t, t, e_text) → ε_pred                      │  │
│  │    2. x̂₀ = (x_t - √(1-ᾱ_t)·ε_pred) / √ᾱ_t             │  │
│  │    3. L_bg = ||x̂₀⊙(1-M) - z_orig⊙(1-M)||²              │  │
│  │    4. L_sc = -mean(attn_map[area mask])                   │  │
│  │    5. L = coef·L_bg + L_sc                               │  │
│  │    6. x_t ← x_t - lr·∇_x_t(L)  (+momentum)              │  │
│  └────────────────────────────────────────────────────────── ┘  │
│                                                                 │
│  ┌─── [Jika t < 1000(1-γ)] ─────────────────────────────────┐  │
│  │  Re-inject noise dari gambar asli ke area mask:           │  │
│  │  x_t = x_t⊙(1-M) + add_noise(z_orig, t)⊙M               │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  DDIM Step: x_{t-1} = DDIM(x_t, ε_pred, t)                    │
└─────────────────────────────────────────────────────────────────┘


TAHAP 4 — DEKODING OUTPUT
┌──────────────────────────────────────────────┐
│  x_0 ∈ R^{4×64×64}                          │
│       ↓ VAE Decoder                          │
│  Gambar RGB ∈ [0,1]^{512×512×3}             │
│       ↓ numpy_to_pil()                       │
│  image_list: List[PIL.Image]                 │
└──────────────────────────────────────────────┘


TAHAP 5 — EVALUASI & SIMPAN [BARU]
┌──────────────────────────────────────────────────────────────┐
│  evaluate_results(image_list, original, mask, prompt, ...)   │
│                                                              │
│  ├── CLIP Score  : cos_sim(CLIP_img, CLIP_text) × 100        │
│  ├── NIMA Score  : MOS = Σ(k × P(rating=k)), k∈[1,10]       │
│  ├── SSIM        : structural_similarity(orig, gen, bg_only) │
│  └── LPIPS       : AlexNet perceptual dist (bg_only)         │
│                                                              │
│  Output:                                                     │
│  ├── generated_{config}_{i}.png  ← Gambar hasil murni        │
│  └── eval_{config}_{timestamp}.json  ← Skor metrik          │
└──────────────────────────────────────────────────────────────┘
```

---

## 7. Penambahan Modul Evaluasi (Kontribusi Penelitian)

Salah satu kontribusi utama penelitian ini adalah **penambahan sistem evaluasi kuantitatif otomatis** yang diintegrasikan ke dalam pipeline PILOT yang sudah ada, **tanpa merombak satu baris pun dari kode dasar**.

### 7.1 `utils/evaluation.py` — Modul Evaluasi Baru

File baru sepenuhnya (~380 baris) yang berisi:

```
utils/evaluation.py
├── _load_clip()              ← lazy-load open_clip ViT-B-32
├── _load_lpips()             ← lazy-load LPIPS AlexNet
├── _load_nima()              ← lazy-load MobileNetV2 + custom head
│
├── compute_clip_score()      ← kalkulasi CLIP Score
├── compute_nima_score()      ← kalkulasi NIMA MOS
├── compute_ssim_non_mask()   ← kalkulasi SSIM pada background
├── compute_lpips_non_mask()  ← kalkulasi LPIPS pada background
│
└── evaluate_results()        ← fungsi utama (entry point)
```

**Desain prinsip:**

1. **Lazy loading** — Setiap library berat (`open_clip`, `lpips`, `skimage`) hanya dimuat saat fungsinya dipanggil, sehingga tidak memperlambat import module lain.
2. **Error isolation** — Setiap metrik dibungkus `try-except`, sehingga jika satu metrik gagal (misal library belum di-install), metrik lain tetap berjalan.
3. **Zero coupling** — Modul ini tidak mengimpor apapun dari `pipeline/` atau `models/`, sehingga bisa dijalankan secara independen.

---

### 7.2 Modifikasi `run_example.py`

Hanya **10 baris** yang ditambahkan dari total 244 baris, terletak tepat setelah `image_list = pipe(...)` selesai:

```python
# ── Evaluation (CLIP, NIMA, SSIM non-mask, LPIPS non-mask) ──────────────────
config_name = os.path.splitext(os.path.basename(args.config_file))[0]
evaluate_results(
    image_list=image_list,
    original_image=image,
    mask_image=mask_image,
    prompt=config.prompt,
    output_path=config.output_path,
    config_name=config_name,
)
```

Serta penambahan satu baris import di bagian atas:

```python
from utils.evaluation import evaluate_results
```

**Tidak ada perubahan** pada logika pipeline, parameter model, atau visualisasi.

---

### 7.3 Update `requirements.txt`

Ditambahkan 5 baris di bagian bawah:

```
# ── Evaluation dependencies ──────────────────────────────────────────────────
open-clip-torch      # CLIP Score
lpips                # Learned Perceptual Image Patch Similarity
scikit-image         # SSIM (structural_similarity)
torchvision          # NIMA (MobileNetV2 backbone)
pandas               # Opsional: ekspor CSV hasil evaluasi
```

---

## 8. Penjelasan Detail Setiap Metrik Evaluasi

### 8.1 CLIP Score

**Referensi:** Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021.

**Definisi:**

$$\text{CLIP Score} = \cos(\mathbf{e}_{img}, \mathbf{e}_{txt}) \times 100$$

Di mana $\mathbf{e}_{img}$ dan $\mathbf{e}_{txt}$ adalah embedding visual dan teks ternormalisasi dari model CLIP ViT-B/32.

**Implementasi:**

```python
img_features = clip_model.encode_image(preprocess(gen_img))
txt_features = clip_model.encode_text(tokenize([prompt]))
score = cosine_similarity(img_features, txt_features) * 100
```

**Interpretasi:**

- Rentang: ~0 – 100
- Semakin tinggi = konten gambar semakin sesuai dengan teks prompt
- Berguna untuk mengevaluasi seberapa baik motif yang dihasilkan sesuai dengan deskripsi batik dalam prompt

---

### 8.2 NIMA Score (Neural Image Assessment)

**Referensi:** Talebi & Milanfar, "NIMA: Neural Image Assessment", IEEE TIP 2018.

**Definisi:**

Model NIMA memprediksi distribusi probabilitas atas 10 tingkat kualitas (1–10). Skor akhir adalah Mean Opinion Score (MOS):

$$\text{NIMA MOS} = \sum_{k=1}^{10} k \cdot P(\text{rating} = k)$$

**Arsitektur model:**

```
MobileNetV2 (backbone ImageNet)
    ↓
AdaptiveAvgPool2d(1)
    ↓
Dropout(0.75) → Linear(1280, 10) → Softmax
    ↓
Distribusi probabilitas rating [P(1), P(2), ..., P(10)]
    ↓
MOS = Σ(k × P(k))
```

**Interpretasi:**

- Rentang: 1 – 10
- Semakin tinggi = kualitas estetika gambar lebih baik
- Berguna untuk menilai apakah hasil inpainting terlihat natural dan berkualitas secara visual

> **⚠️ Catatan:** Untuk hasil optimal, unduh pretrained weights NIMA dari [truskovskiyk/nima.pytorch](https://github.com/truskovskiyk/nima.pytorch) dan simpan di `models/nima_mobilenet.pth`. Tanpa weights khusus, NIMA menggunakan inisialisasi ImageNet.

---

### 8.3 SSIM pada Area Non-Mask (Background)

**Referensi:** Wang et al., "Image Quality Assessment: From Error Visibility to Structural Similarity", IEEE TIP 2004.

**Definisi:**

SSIM mengukur kemiripan struktural antara dua gambar berdasarkan tiga komponen:

$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$

Pada penelitian ini, SSIM dihitung **hanya pada piksel background** (area di luar mask):

$$\text{SSIM}_{bg} = \text{SSIM}(\mathbf{x}_{orig} \odot (1-M),\; \mathbf{x}_{gen} \odot (1-M))$$

**Implementasi:**

```python
bg_mask = (mask_np < 0.5)              # 1 = background, 0 = area edit
orig_masked = orig_np * bg_mask        # Sembunyikan area mask
gen_masked  = gen_np  * bg_mask
ssim_score  = mean([ssim(orig_masked[:,:,c], gen_masked[:,:,c])
                    for c in range(3)])
```

**Interpretasi:**

- Rentang: -1 – 1 (dalam praktik 0 – 1 untuk gambar natural)
- Semakin tinggi = background lebih terjaga / pipeline tidak merusak area di luar mask
- Nilai ideal mendekati 1.0

---

### 8.4 LPIPS pada Area Non-Mask (Background)

**Referensi:** Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric", CVPR 2018.

**Definisi:**

LPIPS mengukur jarak perseptual menggunakan fitur mendalam dari jaringan AlexNet. Berbeda dengan SSIM yang menggunakan statistik piksel, LPIPS lebih sensitif terhadap perbedaan tekstur dan struktur level menengah.

$$\text{LPIPS} = \sum_l \frac{1}{H_l W_l} \sum_{h,w} \left\| w_l \odot (\hat{y}_{hw}^l - y_{hw}^l) \right\|_2^2$$

Pada penelitian ini, dihitung hanya pada area background:

```python
orig_bg = orig_tensor * bg_mask   # Nol-kan area inpainting
gen_bg  = gen_tensor  * bg_mask
lpips_score = loss_fn(orig_bg, gen_bg)
```

**Interpretasi:**

- Rentang: 0 – 1
- Semakin rendah = background lebih terjaga secara perseptual
- Nilai ideal mendekati 0.0

---

### 8.5 Ringkasan Metrik

| Metrik           | Mengukur                         | Area                  | Rentang | Ideal            |
| ---------------- | -------------------------------- | --------------------- | ------- | ---------------- |
| CLIP Score       | Kesesuaian teks-gambar           | Seluruh gambar hasil  | ~0–100  | Semakin tinggi ↑ |
| NIMA Score       | Kualitas estetika visual         | Seluruh gambar hasil  | 1–10    | Semakin tinggi ↑ |
| SSIM (non-mask)  | Preservasi struktur background   | Hanya area background | -1–1    | Semakin tinggi ↑ |
| LPIPS (non-mask) | Preservasi perseptual background | Hanya area background | 0–1     | Semakin rendah ↓ |

---

## 9. Output yang Dihasilkan

Setiap kali `run_example.py` dijalankan, file berikut disimpan di `output_path` (default: `outputs/`):

```
outputs/
├── seed100_step100.png
│   └── Gambar komposit untuk laporan:
│       [Original+Mask] | [Hasil Inpainting]
│
├── generated_t2i_step100_0.png         [BARU]
│   └── Gambar hasil inpainting murni (tanpa komposit)
│       Digunakan sebagai input metrik evaluasi
│
└── eval_t2i_step100_20260225_143022.json   [BARU]
    └── Hasil semua metrik dalam format JSON
```

**Contoh isi file JSON:**

```json
[
  {
    "image_index": 0,
    "config": "t2i_step100",
    "prompt": "traditional batik ornament, golden mandala pattern...",
    "clip_score": 28.4231,
    "nima_score": 5.1823,
    "ssim_non_mask": 0.9712,
    "lpips_non_mask": 0.0341
  }
]
```

**Contoh output terminal:**

```
============================================================
  PILOT Evaluation — Running Metrics
============================================================

[Image 1/1]
  CLIP Score         : 28.4231  (higher is better, ~0-100)
  NIMA Score         : 5.1823   (higher is better, 1-10)
  SSIM (non-mask)    : 0.9712   (higher is better, -1 to 1)
  LPIPS (non-mask)   : 0.0341   (lower is better, 0-1)

  Generated image saved: outputs/generated_t2i_step100_0.png

============================================================
  Evaluation complete. Results saved to:
  outputs/eval_t2i_step100_20260225_143022.json
============================================================
```

---

## 10. Cara Menjalankan Program

### 10.1 Persiapan Lingkungan

```bash
# 1. Clone repositori
git clone https://github.com/wempy-aditya/PILOT-BATIK.git
cd PILOT-BATIK

# 2. Buat dan aktifkan environment conda
conda create -n pilot python==3.9
conda activate pilot

# 3. Install PyTorch (sesuaikan dengan versi CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Install semua dependensi
pip install -r requirements.txt
```

### 10.2 Unduh Model yang Diperlukan

```bash
# Unduh dari Hugging Face (bisa menggunakan git-lfs atau huggingface-cli)

# 1. Stable Diffusion v1.5 (wajib)
# https://huggingface.co/runwayml/stable-diffusion-v1-5
# Simpan di: ./runwayml/stable-diffusion-v1-5/

# 2. ControlNet (opsional, untuk mode spatial)
# https://huggingface.co/lllyasviel/sd-controlnet-scribble
# Simpan di: ./runwayml/sd-controlnet-scribble/

# 3. IP-Adapter (opsional, untuk mode referensi gambar)
# https://huggingface.co/h94/IP-Adapter
# Simpan di: ./runwayml/ip_adapter/v1-5/ip-adapter_sd15_light.bin

# 4. NIMA weights (opsional, untuk metrik NIMA yang akurat)
# https://github.com/truskovskiyk/nima.pytorch
# Simpan di: ./models/nima_mobilenet.pth
```

### 10.3 Siapkan Data Input

```bash
# Tempatkan gambar batik di folder data/
# Format: RGB PNG atau JPG, ukuran bebas (akan diresize ke 512×512)

data/
├── gambar_batik.jpg    # Gambar batik input
└── mask_batik.png      # Mask area yang ingin diedit
                        # Putih (255,255,255) = area edit
                        # Hitam (0,0,0)       = background
```

### 10.4 Konfigurasi Eksperimen

Edit file YAML sesuai eksperimen:

```bash
# Salin template konfigurasi
cp configs/t2i_step100.yaml configs/batik_experiment_1.yaml

# Edit parameter utama
# - model_path: path ke direktori model
# - input_image: path ke gambar batik
# - mask_image: path ke mask
# - prompt: deskripsi motif yang diinginkan
```

### 10.5 Jalankan Eksperimen

```bash
# Mode 1: Text-guided inpainting (paling dasar)
python run_example.py --config_file configs/t2i_step100.yaml

# Mode 2: Dengan spatial control (ControlNet)
python run_example.py --config_file configs/controlnet_step100.yaml

# Mode 3: Dengan referensi gambar (IP-Adapter)
python run_example.py --config_file configs/ipa_step100.yaml

# Mode 4: Kombinasi ControlNet + IP-Adapter
python run_example.py --config_file configs/ipa_controlnet_step100.yaml
```

---

## 11. Konfigurasi Parameter

### 11.1 Parameter Wajib

| Parameter         | Tipe | Contoh                    | Keterangan                      |
| ----------------- | ---- | ------------------------- | ------------------------------- |
| `model_path`      | str  | `"runwayml"`              | Path direktori semua model      |
| `model_id`        | str  | `"stable-diffusion-v1-5"` | Subfolder model base            |
| `input_image`     | str  | `"data/batik.jpg"`        | Path gambar input               |
| `mask_image`      | str  | `"data/mask.png"`         | Path mask (hitam/putih)         |
| `prompt`          | str  | `"batik kawung..."`       | Deskripsi motif yang diinginkan |
| `negative_prompt` | str  | `"blurry, distorted"`     | Hal yang tidak diinginkan       |
| `output_path`     | str  | `"outputs"`               | Direktori simpan hasil          |

### 11.2 Parameter Generasi

| Parameter | Default | Keterangan                                                  |
| --------- | ------- | ----------------------------------------------------------- |
| `W` / `H` | `512`   | Dimensi output dalam piksel                                 |
| `num`     | `1`     | Jumlah gambar yang dihasilkan                               |
| `seed`    | `100`   | Random seed (untuk reproduksibilitas)                       |
| `cfg`     | `7.5`   | Classifier-free guidance scale (1=bebas, 30=sangat terikat) |
| `step`    | `100`   | Jumlah langkah denoising                                    |
| `fp16`    | `true`  | Gunakan FP16 untuk hemat VRAM                               |

### 11.3 Parameter Optimasi PILOT (Inti Metode)

| Parameter          | Default    | Pengaruh                                                                      |
| ------------------ | ---------- | ----------------------------------------------------------------------------- |
| `op_interval`      | `10`       | Seberapa sering optimasi dilakukan. Lebih kecil = lebih presisi, lebih lambat |
| `num_gradient_ops` | `10`       | Iterasi gradient per langkah. Lebih besar = lebih stabil, lebih lambat        |
| `gamma`            | `1.0`      | Proporsi timestep yang mendapat optimasi (0.5 = hanya 50% awal)               |
| `lr`               | `0.025`    | Learning rate optimasi laten                                                  |
| `lr_warmup`        | `0.007`    | Learning rate warmup awal                                                     |
| `lr_f`             | `"exp"`    | Jadwal learning rate: `"exp"` atau `"linear"`                                 |
| `coef`             | `150`      | Bobot background loss. Lebih besar = background lebih terjaga                 |
| `coef_f`           | `"linear"` | Jadwal koefisien background loss                                              |
| `momentum`         | `0.7`      | Momentum gradient update (0=tanpa momentum)                                   |

---

## 12. Dependensi dan Lingkungan

### 12.1 Spesifikasi Lingkungan yang Direkomendasikan

| Komponen    | Rekomendasi                                 |
| ----------- | ------------------------------------------- |
| **OS**      | Ubuntu 20.04 / Windows 10+                  |
| **GPU**     | NVIDIA dengan VRAM ≥ 8GB (RTX 3070 ke atas) |
| **CUDA**    | 11.8 atau 12.x                              |
| **Python**  | 3.9                                         |
| **PyTorch** | 2.0+ dengan CUDA 11.8                       |

### 12.2 Seluruh Dependensi (`requirements.txt`)

```
# Dependensi Base (bawaan PILOT)
diffusers==0.29.2
einops
numpy==1.26.3
Pillow==10.2.0
PyYAML
safetensors
tqdm
transformers==4.37.2
opencv-python
omegaconf
controlnet-aux==0.0.7
accelerate==0.26.1

# Dependensi Evaluasi (ditambahkan penelitian ini)
open-clip-torch      # CLIP Score
lpips                # LPIPS
scikit-image         # SSIM
torchvision          # NIMA backbone
pandas               # Ekspor CSV (opsional)
```

---

## 13. Rencana Eksperimen & Perbandingan

### 13.1 Skenario Eksperimen

Penelitian ini akan menjalankan dua kelompok eksperimen utama:

**Kelompok A — Eksperimen Bawaan PILOT (Baseline)**

| ID  | Config                    | Input              | Prompt             |
| --- | ------------------------- | ------------------ | ------------------ |
| A1  | `t2i_step100.yaml`        | Gambar bawaan repo | Prompt bawaan repo |
| A2  | `controlnet_step100.yaml` | Gambar bawaan repo | Prompt bawaan repo |
| A3  | `ipa_step100.yaml`        | Gambar bawaan repo | Prompt bawaan repo |

**Kelompok B — Eksperimen Batik (Kontribusi Utama)**

| ID  | Config                    | Input                          | Prompt             |
| --- | ------------------------- | ------------------------------ | ------------------ |
| B1  | `t2i_step100.yaml`        | Gambar motif batik             | Prompt motif batik |
| B2  | `controlnet_step100.yaml` | Gambar motif batik + scribble  | Prompt motif batik |
| B3  | `ipa_step100.yaml`        | Gambar motif batik + ref batik | Prompt motif batik |

### 13.2 Tabel Perbandingan (Template)

| Eksperimen               | CLIP Score ↑ | NIMA ↑ | SSIM (bg) ↑ | LPIPS (bg) ↓ |
| ------------------------ | ------------ | ------ | ----------- | ------------ |
| A1 (baseline t2i)        | —            | —      | —           | —            |
| A2 (baseline controlnet) | —            | —      | —           | —            |
| A3 (baseline ipa)        | —            | —      | —           | —            |
| B1 (batik t2i)           | —            | —      | —           | —            |
| B2 (batik controlnet)    | —            | —      | —           | —            |
| B3 (batik ipa)           | —            | —      | —           | —            |

### 13.3 Analisis yang Direncanakan

1. **Perbandingan antar-kelompok:** Apakah pipeline PILOT dapat mempertahankan performanya saat domain beralih ke gambar batik?
2. **Pengaruh kondisi tambahan:** Apakah penggunaan ControlNet atau IP-Adapter meningkatkan metrik secara konsisten?
3. **Trade-off koherensi vs. fidelitas:** Apakah SSIM yang tinggi (background terjaga) selalu berkorelasi dengan CLIP Score yang tinggi (konten sesuai)?
4. **Analisis kualitatif:** Inspeksi visual terhadap pola, warna, dan keselarasan motif batik yang dihasilkan.

---

## 14. Referensi

```bibtex
@article{pan2024coherent,
  title     = {Coherent and Multi-modality Image Inpainting via Latent Space Optimization},
  author    = {Pan, Lingzhi and Zhang, Tong and Chen, Bingyuan and
               Zhou, Qi and Ke, Wei and Susstrunk, Sabine and Salzmann, Mathieu},
  journal   = {arXiv preprint arXiv:2407.08019},
  year      = {2024}
}

@inproceedings{radford2021learning,
  title     = {Learning Transferable Visual Models From Natural Language Supervision},
  author    = {Radford, Alec and others},
  booktitle = {ICML},
  year      = {2021}
}

@article{talebi2018nima,
  title   = {NIMA: Neural Image Assessment},
  author  = {Talebi, Hossein and Milanfar, Peyman},
  journal = {IEEE Transactions on Image Processing},
  year    = {2018}
}

@article{wang2004image,
  title   = {Image Quality Assessment: From Error Visibility to Structural Similarity},
  author  = {Wang, Zhou and others},
  journal = {IEEE Transactions on Image Processing},
  year    = {2004}
}

@inproceedings{zhang2018perceptual,
  title     = {The Unreasonable Effectiveness of Deep Features as a Perceptual Metric},
  author    = {Zhang, Richard and others},
  booktitle = {CVPR},
  year      = {2018}
}
```

---

_Dokumentasi ini dibuat sebagai bagian dari laporan penelitian proyek PILOT-BATIK._  
_Terakhir diperbarui: 25 Februari 2026_
