import glob
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet

# ---------------- 1. KONFIGURACIJA ----------------
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
MODEL_PATH = BASE_DIR / "best_segresnet.pt"
CASE_DIR = BASE_DIR / "test1"
PRED_DIR = BASE_DIR / "preds"
PRED_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = (64, 64, 64)
SW_BATCH_SIZE = 8
INIT_FILTERS = 16
IN_CHANNELS = 4
OUT_CHANNELS = 4  # NUM_CLASSES = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- 2. UČITAVANJE MODELA ----------------
model = SegResNet(
    spatial_dims=3,
    init_filters=INIT_FILTERS,
    in_channels=IN_CHANNELS,
    out_channels=OUT_CHANNELS,
    dropout_prob=0.2,
).to(device)

loaded = torch.load(MODEL_PATH, map_location=device, weights_only=False)
if isinstance(loaded, dict):
    state_dict = loaded.get("model_state_dict") or loaded.get("state_dict") or loaded
else:
    state_dict = loaded

cleaned_state_dict = {
    (k[7:] if k.startswith("module.") else k): v
    for k, v in state_dict.items()
}

model.load_state_dict(cleaned_state_dict)
model.eval()
print(f"Model učitan iz '{MODEL_PATH}' na uređaj: {device}")


# ---------------- 3. UČITAVANJE I NORMALIZACIJA FOLDERA ----------------
def _find_modality_in_dir(case_dir: Path, mod: str) -> str:
    patterns = [
        str(case_dir / f"*_{mod}.nii.gz"),
        str(case_dir / f"*_{mod}.nii"),
        str(case_dir / f"*{mod}*.nii.gz"),
        str(case_dir / f"*{mod}*.nii"),
    ]
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Nedostaje modalitet '{mod}' u folderu: {case_dir}")


def load_and_normalize_folder(case_dir: Path):
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Folder ne postoji: {case_dir}")

    cid = case_dir.resolve().name
    modalities = ["t1", "t1ce", "t2", "flair"]
    chans = []
    ref_nii = None

    for m in modalities:
        mod_path = _find_modality_in_dir(case_dir, m)
        nii = nib.load(mod_path)
        if ref_nii is None:
            ref_nii = nii
        img = nii.get_fdata().astype(np.float32)
        mask_nz = img > 0
        if np.any(mask_nz):
            img = np.where(mask_nz, (img - img[mask_nz].mean()) / (img[mask_nz].std() + 1e-8), 0.0)
        chans.append(img)

    tensor = torch.from_numpy(np.stack(chans, axis=0)[None, ...])  # (1, 4, H, W, D)
    return cid, tensor, ref_nii.affine, ref_nii.header


# ---------------- 4. INFERENCIJA I SPREMANJE ----------------
with torch.no_grad():
    cid, imgs, affine, header = load_and_normalize_folder(CASE_DIR)
    imgs = imgs.to(device, non_blocking=True)

    if device.type == "cuda":
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = sliding_window_inference(
                inputs=imgs,
                roi_size=PATCH_SIZE,
                sw_batch_size=SW_BATCH_SIZE,
                predictor=model,
                overlap=0.5,
                mode="gaussian",
            )
    else:
        logits = sliding_window_inference(
            inputs=imgs,
            roi_size=PATCH_SIZE,
            sw_batch_size=2,
            predictor=model,
            overlap=0.5,
            mode="gaussian",
        )

    pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    pred_mask[pred_mask == 3] = 4

    out_file = PRED_DIR / "_seg.nii.gz"
    out_img = nib.Nifti1Image(pred_mask, affine=affine, header=header)
    out_img.set_data_dtype(np.uint8)
    nib.save(out_img, str(out_file))

print(f"Inferencija gotova. Maska spremljena u: {out_file}")
