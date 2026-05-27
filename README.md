# 🎨 Image Colorization with U-Net and GAN

Automatic colorization of black & white photos using a 3-stage GAN pipeline built from scratch in PyTorch.

**[LIVE →](https://ugh-colorizer.streamlit.app/)**

---

## How It Works

The model takes the **L (lightness) channel** of an image as input and predicts the **a and b (color) channels** using the LAB color space — keeping grayscale structure intact while adding realistic color.

---

## 3-Stage Training Pipeline

### Stage 1 — Baseline U-Net GAN `(100 epochs)`

A custom U-Net generator (8 downsampling levels, 64 filters) paired with a PatchGAN discriminator trained adversarially from scratch.

- **Loss:** GAN (vanilla BCE) + λ·L1 (λ=100)
- **Optimizer:** Adam (lr=2e-4, β₁=0.5)
- **Result:** Working colorization baseline — but slow to converge and colors are muted

### Stage 2 — ResNet18-UNet Pre-training `(20 epochs, L1 only)`

Replaced the generator with a ResNet18-backed DynamicUNet (fastai). Pretrained with L1 loss only — no discriminator — so the generator learns colorization in a stable, supervised way before adversarial training begins.

- **Backbone:** ResNet18 pretrained on ImageNet (conv1 modified for 1-channel input)
- **Loss:** L1 only
- **Result:** Generator already produces reasonable colors before seeing a single adversarial signal

### Stage 3 — Fine-tuning with GAN `(20 epochs)`

Loaded the pre-trained generator from Stage 2 into the full GAN setup with a fresh discriminator. Because the generator already knows how to colorize, the GAN converges in 20 epochs instead of 100.

- **Loss:** GAN + λ·L1 (λ=100)
- **AMP:** `autocast` + separate `GradScaler` for G and D
- **Final losses:** `loss_D: 0.611` · `loss_G_L1: 7.087` · `loss_G: 8.077`
- **Result:** Vivid, perceptually realistic colorization

---

## Architecture

| Component     | Details                                                                  |
| ------------- | ------------------------------------------------------------------------ |
| Generator     | ResNet18-UNet (encoder: ResNet18, decoder: DynamicUnet skip connections) |
| Discriminator | PatchGAN — 3 downsampling layers, 64 filters, outputs 30×30 patch map    |
| Input         | L channel `[1, 1, 256, 256]` normalized to `[-1, 1]`                     |
| Output        | ab channels `[1, 2, 256, 256]` → combined with L → LAB → RGB             |
| Loss          | BCE (GAN) + L1 weighted at λ=100                                         |
| Optimizer     | Adam (lr=2e-4, β₁=0.5, β₂=0.999)                                         |

---

## Training Details

|               | Stage 1      | Stage 2       | Stage 3       |
| ------------- | ------------ | ------------- | ------------- |
| Generator     | Custom U-Net | ResNet18-UNet | ResNet18-UNet |
| Discriminator | PatchGAN     | None          | PatchGAN      |
| Epochs        | 100          | 20            | 20            |
| Loss          | GAN + L1     | L1 only       | GAN + L1      |
| AMP           | No           | Yes           | Yes           |
| Time          | ~7 hrs       | ~45 min       | ~1.5 hrs      |

**Hardware:** NVIDIA RTX 3050 4GB · Batch size: 16 · Image size: 256×256
**Dataset:** COCO Sample — 8,000 train / 2,000 val

---

## Saved Weights

| File                           | Description                                   |
| ------------------------------ | --------------------------------------------- |
| `baseline_epoch_100.pt`        | Stage 1 final — plain U-Net GAN               |
| `res18-unet.pt`                | Stage 2 — pre-trained ResNet18-UNet generator |
| `finetune_epoch_5/10/15/20.pt` | Stage 3 checkpoints                           |
| `final_model.pt`               | **Final model — use this for inference**      |

---

## Inference

```python
net_G = build_res_unet(n_input=1, n_output=2, size=256)
model = MainModel(net_G=net_G)
model.load_state_dict(torch.load("final_model.pt", map_location=device))
model.eval()

colorize_image("your_image.jpg", model)
# saves → colorized_result.png
```

---

## Stack

`PyTorch` · `fastai` · `torchvision` · `scikit-image` · `Streamlit`

---
