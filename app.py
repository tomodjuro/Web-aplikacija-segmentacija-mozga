import spaces  # Must be imported before torch

import tempfile
from pathlib import Path

import gradio as gr
import nibabel as nib
import numpy as np
import torch

from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "best_segresnet.pt"

PATCH_SIZE = (64, 64, 64)
SW_BATCH_SIZE = 2

INIT_FILTERS = 16
IN_CHANNELS = 4
OUT_CHANNELS = 4

MODALITIES = ["t1", "t1ce", "t2", "flair"]

# Maximum dimensions of each MRI panel
PANEL_MAX_WIDTH = 350
PANEL_MAX_HEIGHT = 230

# Space between panels
PANEL_GAP = 8

# RGB tumor colors
SEG_COLORS = {
    1: np.array([255, 48, 48], dtype=np.float32),
    2: np.array([0, 255, 85], dtype=np.float32),
    4: np.array([37, 133, 255], dtype=np.float32),
}

_CACHED_MODEL = None
_CACHED_MODEL_PATH = None


# ============================================================
# MODEL LOADING
# ============================================================

def get_model(model_path_str=None):

    global _CACHED_MODEL
    global _CACHED_MODEL_PATH

    model_path = (
        Path(model_path_str)
        if model_path_str
        else DEFAULT_MODEL_PATH
    )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint nije pronađen: {model_path}"
        )

    device = torch.device("cuda")

    if (
        _CACHED_MODEL is not None
        and _CACHED_MODEL_PATH == model_path.resolve()
    ):
        return _CACHED_MODEL, device

    if _CACHED_MODEL is not None:
        del _CACHED_MODEL
        _CACHED_MODEL = None

    torch.cuda.empty_cache()

    model = SegResNet(
        spatial_dims=3,
        init_filters=INIT_FILTERS,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        dropout_prob=0.2,
    )

    loaded = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(loaded, dict):
        state_dict = (
            loaded.get("model_state_dict")
            or loaded.get("state_dict")
            or loaded
        )
    else:
        state_dict = loaded

    cleaned_state_dict = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    model.load_state_dict(cleaned_state_dict)

    model = model.to(device)
    model.eval()

    _CACHED_MODEL = model
    _CACHED_MODEL_PATH = model_path.resolve()

    return model, device


# ============================================================
# FILE MATCHING
# ============================================================

def _match_modality(file_paths, mod):

    target = f"_{mod.lower()}"

    for p in sorted(file_paths):

        name = p.name.lower()

        if name.endswith(".nii.gz"):
            stem = name[:-7]

        elif name.endswith(".nii"):
            stem = name[:-4]

        else:
            continue

        if stem.endswith(target):
            return p

    raise FileNotFoundError(
        f"Nedostaje modalitet '{mod}'."
    )


# ============================================================
# NIFTI LOADING
# ============================================================

def load_nifti_canonical(path):

    nii = nib.as_closest_canonical(
        nib.load(str(path))
    )

    data = nii.get_fdata(dtype=np.float32)

    if data.ndim != 3:
        raise ValueError(
            f"{path.name} mora biti 3D volumen."
        )

    if not np.all(np.isfinite(data)):
        raise ValueError(
            f"{path.name} sadrži neispravne vrijednosti."
        )

    return data, nii.affine


# ============================================================
# IMAGE NORMALIZATION
# ============================================================

def normalize_image(img):

    low, high = np.percentile(
        img,
        [1, 99],
    )

    if high <= low:
        high = low + 1.0

    normalized = np.clip(
        (img - low) / (high - low),
        0.0,
        1.0,
    )

    return np.round(
        normalized * 255
    ).astype(np.uint8)


# ============================================================
# PRECOMPUTED MRI SLICE CACHE
# ============================================================

def create_display_cache(volumes):

    """
    Normalize and resize all MRI slices only once.

    This avoids recalculating percentiles and normalization
    every time the user moves the Z-slice slider.
    """

    display_cache = {}

    ref_volume = volumes[MODALITIES[0]]

    depth = ref_volume.shape[2]

    original_height = ref_volume.shape[1]
    original_width = ref_volume.shape[0]

    # Keep original aspect ratio
    scale = min(
        1.0,
        PANEL_MAX_WIDTH / original_width,
        PANEL_MAX_HEIGHT / original_height,
    )

    display_width = max(
        1,
        int(round(original_width * scale)),
    )

    display_height = max(
        1,
        int(round(original_height * scale)),
    )

    # Precompute nearest-neighbor sampling indices
    y_idx = np.linspace(
        0,
        original_height - 1,
        display_height,
    ).round().astype(np.intp)

    x_idx = np.linspace(
        0,
        original_width - 1,
        display_width,
    ).round().astype(np.intp)

    # Precompute every MRI modality
    for mod in MODALITIES:

        volume = volumes[mod]

        cached_slices = np.empty(
            (
                depth,
                display_height,
                display_width,
            ),
            dtype=np.uint8,
        )

        for z in range(depth):

            # Original orientation used by the viewer
            img = volume[:, :, z].T

            # Normalize only once
            normalized = normalize_image(img)

            # Downsample for laptop-friendly display
            resized = normalized[
                np.ix_(y_idx, x_idx)
            ]

            cached_slices[z] = resized

        display_cache[mod] = cached_slices

    return {
        "images": display_cache,
        "x_idx": x_idx,
        "y_idx": y_idx,
        "height": display_height,
        "width": display_width,
        "max_slice": depth - 1,
    }


# ============================================================
# FAST RGB MRI RENDERING
# ============================================================

def render_slice_image(
    display_cache,
    segmentation,
    slice_idx,
    show_overlay,
    opacity,
):

    images = display_cache["images"]

    x_idx = display_cache["x_idx"]
    y_idx = display_cache["y_idx"]

    height = display_cache["height"]
    width = display_cache["width"]

    rendered_images = []

    for mod in MODALITIES:

        gray = images[mod][slice_idx]

        rgb = np.repeat(
            gray[:, :, None],
            3,
            axis=2,
        ).astype(np.float32)

        if show_overlay and segmentation is not None:

            seg_slice = segmentation[
                :, :, slice_idx
            ].T

            seg_small = seg_slice[
                np.ix_(y_idx, x_idx)
            ]

            for label, color in SEG_COLORS.items():

                mask = seg_small == label

                if np.any(mask):

                    rgb[mask] = (
                        rgb[mask] * (1.0 - opacity)
                        + color * opacity
                    )

        rendered_images.append(
            np.clip(rgb, 0, 255).astype(np.uint8)
        )

    # Build 2x2 canvas
    canvas_height = height * 2 + PANEL_GAP
    canvas_width = width * 2 + PANEL_GAP

    canvas = np.zeros(
        (canvas_height, canvas_width, 3),
        dtype=np.uint8,
    )

    canvas[:height, :width] = rendered_images[0]

    canvas[
        :height,
        width + PANEL_GAP:
    ] = rendered_images[1]

    canvas[
        height + PANEL_GAP:,
        :width
    ] = rendered_images[2]

    canvas[
        height + PANEL_GAP:,
        width + PANEL_GAP:
    ] = rendered_images[3]

    return canvas


# ============================================================
# SEGMENTATION PIPELINE
# ============================================================

@spaces.GPU(duration=120)
def run_pipeline(
    uploaded_files,
    show_overlay,
    opacity_pct,
    slice_idx,
):

    if not uploaded_files:
        raise gr.Error(
            "Učitaj sva 4 MRI modaliteta "
            "(*_t1.nii, *_t1ce.nii, *_t2.nii, *_flair.nii)."
        )

    file_paths = [
        Path(
            f.name if hasattr(f, "name") else f
        )
        for f in uploaded_files
    ]

    # --------------------------------------------------------
    # Load MRI modalities
    # --------------------------------------------------------

    mod_paths = {
        mod: _match_modality(file_paths, mod)
        for mod in MODALITIES
    }

    volumes = {}

    ref_shape = None
    ref_affine = None

    for mod in MODALITIES:

        data, affine = load_nifti_canonical(
            mod_paths[mod]
        )

        if ref_shape is None:

            ref_shape = data.shape
            ref_affine = affine

        else:

            if data.shape != ref_shape:
                raise gr.Error(
                    f"Dimenzije za '{mod}' ne odgovaraju ostalim modalitetima."
                )

            if not np.allclose(
                affine,
                ref_affine,
                atol=1e-3,
            ):
                raise gr.Error(
                    f"Afina matrica za '{mod}' se ne podudara."
                )

        volumes[mod] = data

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, device = get_model(None)

    # --------------------------------------------------------
    # Normalize MRI channels for inference
    # --------------------------------------------------------

    chans = []

    for mod in MODALITIES:

        img = volumes[mod].astype(np.float32)

        mask_nz = img > 0

        if np.any(mask_nz):

            mean = img[mask_nz].mean()
            std = img[mask_nz].std()

            img = np.where(
                mask_nz,
                (img - mean) / (std + 1e-8),
                0.0,
            )

        chans.append(img)

    inputs = torch.from_numpy(
        np.stack(chans, axis=0)[None, ...]
    ).to(device)

    # --------------------------------------------------------
    # MONAI sliding window inference
    # --------------------------------------------------------

    with torch.inference_mode():

        if device.type == "cuda":

            with torch.amp.autocast(
                "cuda",
                dtype=torch.float16,
            ):

                logits = sliding_window_inference(
                    inputs=inputs,
                    roi_size=PATCH_SIZE,
                    sw_batch_size=SW_BATCH_SIZE,
                    predictor=model,
                    overlap=0.5,
                    mode="gaussian",
                )

        else:

            logits = sliding_window_inference(
                inputs=inputs,
                roi_size=PATCH_SIZE,
                sw_batch_size=2,
                predictor=model,
                overlap=0.5,
                mode="gaussian",
            )

    # --------------------------------------------------------
    # Create segmentation mask
    # --------------------------------------------------------

    pred_mask = (
        torch.argmax(logits, dim=1)
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    # Convert model class 3 to BraTS label 4
    pred_mask[pred_mask == 3] = 4

    # --------------------------------------------------------
    # Save NIFTI segmentation
    # --------------------------------------------------------

    out_dir = Path(
        tempfile.mkdtemp(prefix="brats_seg_")
    )

    out_file = out_dir / "_seg.nii"

    out_img = nib.Nifti1Image(
        pred_mask,
        affine=ref_affine,
    )

    out_img.set_data_dtype(np.uint8)

    nib.save(
        out_img,
        str(out_file),
    )

    # --------------------------------------------------------
    # Create precomputed display cache
    # --------------------------------------------------------

    display_cache = create_display_cache(volumes)
    max_slice = display_cache["max_slice"]

    clamped_slice = int(
        np.clip(
            slice_idx if slice_idx is not None
            else max_slice // 2,
            0,
            max_slice,
        )
    )

    cache_file = out_dir / "display_cache.npz"
    seg_file = out_dir / "segmentation.npy"

    np.savez_compressed(cache_file, **display_cache["images"])
    np.save(seg_file, pred_mask)

    image = render_slice_image(
        display_cache,
        pred_mask,
        clamped_slice,
        show_overlay,
        opacity_pct / 100.0,
    )

    # Store only lightweight metadata and file paths in gr.State
    state_payload = {
        "cache_path": str(cache_file),
        "seg_path": str(seg_file),
        "x_idx": display_cache["x_idx"].tolist(),
        "y_idx": display_cache["y_idx"].tolist(),
        "height": display_cache["height"],
        "width": display_cache["width"],
        "max_slice": max_slice,
    }

    status = (
        f"Gotovo | Device: {device} | "
        f"Shape: {ref_shape} | "
        f"Classes: {np.unique(pred_mask).tolist()} | "
        f"Output: _seg.nii"
    )

    del inputs
    del logits

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return (
        state_payload,
        image,
        gr.update(
            minimum=0,
            maximum=max_slice,
            value=clamped_slice,
        ),
        str(out_file),
        status,
    )


# ============================================================
# FAST VIEWER UPDATE
# ============================================================

_STATE_ARRAY_CACHE = {}


def _load_state_arrays(state_payload):
    cache_path = state_payload["cache_path"]
    seg_path = state_payload["seg_path"]
    key = (cache_path, seg_path)

    if key in _STATE_ARRAY_CACHE:
        return _STATE_ARRAY_CACHE[key]

    with np.load(cache_path) as npz:
        images = {mod: npz[mod] for mod in MODALITIES}

    display_cache = {
        "images": images,
        "x_idx": state_payload["x_idx"],
        "y_idx": state_payload["y_idx"],
        "height": state_payload["height"],
        "width": state_payload["width"],
        "max_slice": state_payload["max_slice"],
    }
    segmentation = np.load(seg_path)

    _STATE_ARRAY_CACHE[key] = (display_cache, segmentation)
    return display_cache, segmentation

def update_viewer(
    state_payload,
    slice_idx,
    show_overlay,
    opacity_pct,
):

    if not state_payload:
        return None

    display_cache, segmentation = _load_state_arrays(state_payload)
    # Restore integer index arrays for fast slicing
    display_cache = {
        **display_cache,
        "x_idx": np.asarray(display_cache["x_idx"], dtype=np.intp),
        "y_idx": np.asarray(display_cache["y_idx"], dtype=np.intp),
    }

    max_slice = state_payload["max_slice"]

    clamped = int(
        np.clip(
            slice_idx,
            0,
            max_slice,
        )
    )

    return render_slice_image(
        display_cache,
        segmentation,
        clamped,
        show_overlay,
        opacity_pct / 100.0,
    )

# ============================================================
# DARK THEME CSS
# ============================================================

CUSTOM_CSS = """
:root {
    color-scheme: dark;
}

body {
    background: #0b1120 !important;
}

.gradio-container {
    background: #0b1120 !important;
    color: #f8fafc !important;
    max-width: 1100px !important;
}

/* Kartice i blokovi */
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel {
    background: #151e2e !important;
    border-color: #334155 !important;
}

/* Sve oznake i labele (uključujući slider badgeve i file label) */
.gradio-container label,
.gradio-container label span,
.gradio-container label[data-testid="block-label"],
.gradio-container [data-testid="block-label"],
.gradio-container .gr-file label,
.gradio-container .label,
.gradio-container .panel-title,
.gradio-container .block-title,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4 {
    color: #f8fafc !important;
}

.gradio-container label[data-testid="block-label"],
.gradio-container [data-testid="block-label"],
.gradio-container .gr-file label {
    background-color: #1e293b !important;
    border: 1px solid #475569 !important;
    border-radius: 6px !important;
    padding: 4px 10px !important;
}

.gradio-container label[data-testid="block-label"] svg,
.gradio-container [data-testid="block-label"] svg,
.gradio-container .gr-file label svg {
    color: #f8fafc !important;
    fill: #f8fafc !important;
}

/* Tekst i status polja */
.gradio-container p,
.gradio-container .secondary,
.gradio-container .info,
.gradio-container span {
    color: #f8fafc !important;
}

/* Polja za unos (uključujući numeričke unose slidera i Status) */
.gradio-container input,
.gradio-container textarea,
.gradio-container input[type="number"] {
    background: #1e293b !important;
    color: #f8fafc !important;
    border: 1px solid #475569 !important;
}

/* Gumbi */
.gradio-container button {
    border-radius: 9px !important;
}

.gradio-container .primary {
    background: #2563eb !important;
    color: #ffffff !important;
}

/* MRI prikaz */
#viewer-image {
    max-width: 760px;
    margin-left: auto;
    margin-right: auto;
}

#viewer-image img {
    object-fit: contain !important;
}

/* Legenda */
#legend {
    text-align: center;
    background: #1e293b !important;
    border: 1px solid #64748b !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    margin-top: 10px !important;
}

#legend p,
#legend span,
#legend strong {
    margin: 4px 0 !important;
    font-size: 15px !important;
    color: #f8fafc !important;
}

/* Slider oznake */
.gradio-container input[type="range"] {
    accent-color: #3b82f6;
}

/* FIX: Gradio checkbox interaction */
.gradio-container input[type="checkbox"] {
    appearance: auto !important;
    -webkit-appearance: checkbox !important;
    opacity: 1 !important;
    visibility: visible !important;
    pointer-events: auto !important;
    position: relative !important;
    z-index: 10 !important;
    accent-color: #3b82f6 !important;
    cursor: pointer !important;
}

.gradio-container label {
    pointer-events: auto !important;
    cursor: pointer !important;
}

.gradio-container input[type="checkbox"] + span {
    color: #f8fafc !important;
}
#z-slice-slider label,
#z-slice-slider label span,
#z-slice-slider [data-testid="block-label"],
#opacity-slider label,
#opacity-slider label span,
#opacity-slider [data-testid="block-label"] {
    background: #1e293b !important;
    background-color: #1e293b !important;
    color: #f8fafc !important;
    border: 1px solid #475569 !important;
}
#status-box label,
#status-box label span,
#status-box [data-testid="block-label"] {
    background: #1e293b !important;
    background-color: #1e293b !important;
    color: #f8fafc !important;
    border: 1px solid #475569 !important;
}
"""


# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(
    title="BraTS SegResNet Viewer",
    theme=gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
    ),
    css=CUSTOM_CSS,
) as demo:

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    gr.Markdown(
        """
        # 🧠 BraTS Multimodal MRI Viewer

        ### SegResNet 3D Tumor Segmentation

        Učitaj četiri MRI modaliteta i pokreni
        automatsku segmentaciju tumora.
        """
    )

    state = gr.State(value=None)

    # --------------------------------------------------------
    # INPUT FILES
    # --------------------------------------------------------

    with gr.Row():

        nii_files = gr.File(
            label="MRI modaliteti (T1, T1ce, T2, FLAIR)",
            file_count="multiple",
        )

    

    # --------------------------------------------------------
    # RUN BUTTON
    # --------------------------------------------------------

    run_btn = gr.Button(
        "▶ Pokreni segmentaciju",
        variant="primary",
    )

    # --------------------------------------------------------
    # VIEWER CONTROLS
    # --------------------------------------------------------

    with gr.Row():

        slice_slider = gr.Slider(
            label="Z-sloj",
            minimum=0,
            maximum=154,
            value=77,
            step=1,
            scale=4,
            elem_id="z-slice-slider",
        )

        opacity_slider = gr.Slider(
            label="Prozirnost segmentacije (%)",
            minimum=0,
            maximum=100,
            value=45,
            step=1,
            scale=2,
            elem_id="opacity-slider",
        )

        show_overlay = gr.Checkbox(
            label="Prikaži segmentaciju",
            value=True,
            scale=1,
            interactive=True,
        )

    # --------------------------------------------------------
    # MRI 2x2 VIEWER
    # --------------------------------------------------------

    image_out = gr.Image(
        label="MRI pregled - T1 | T1ce | T2 | FLAIR",
        type="numpy",
        height=480,
        width=760,
        interactive=False,
        elem_id="viewer-image",
    )

    # --------------------------------------------------------
    # LEGEND BELOW MRI VIEWER
    # --------------------------------------------------------

    gr.Markdown(
        """
        <div id="legend">
            <p><strong>LEGENDA SEGMENTACIJE</strong></p>
            <p>
                🔴 <strong>NCR / NET (1)</strong>
                &nbsp;&nbsp;
                🟢 <strong>Edema (2)</strong>
                &nbsp;&nbsp;
                🔵 <strong>Enhancing Tumor (4)</strong>
            </p>
        </div>
        """
    )

    # --------------------------------------------------------
    # OUTPUTS
    # --------------------------------------------------------

    with gr.Row():

        download_out = gr.File(
            label="Preuzmi segmentaciju (_seg.nii)",
        )

        status_box = gr.Textbox(
            label="Status",
            interactive=False,
            elem_id="status-box",
        )

    # --------------------------------------------------------
    # RUN SEGMENTATION EVENT
    # --------------------------------------------------------

    run_btn.click(
        fn=run_pipeline,
        inputs=[
            nii_files,
            show_overlay,
            opacity_slider,
            slice_slider,
        ],
        outputs=[
            state,
            image_out,
            slice_slider,
            download_out,
            status_box,
        ],
    )

    # --------------------------------------------------------
    # FAST SLICE AND OVERLAY UPDATES
    # --------------------------------------------------------

    for ctrl in (
        slice_slider,
        show_overlay,
        opacity_slider,
    ):

        ctrl.change(
            fn=update_viewer,
            inputs=[
                state,
                slice_slider,
                show_overlay,
                opacity_slider,
            ],
            outputs=[image_out],
            trigger_mode="always_last",
            show_progress="hidden",
        )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":

    demo.queue()
    demo.launch()