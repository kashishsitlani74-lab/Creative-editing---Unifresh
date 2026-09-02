"""
Creative Editing - Unified
Keeps the original image look by default. No forced white background,
no shadow. Cropping only happens if the user chooses Remove padding.

This is the "Resize / edit images" engine carried over as-is from the
original Atlas tool - same marketplace presets, same resize modes, same
padding/background logic, same output format/quality/DPI controls.

Removed vs. the original:
  - No Excel-links / paste-links input tabs - direct upload only.
  - No custom renaming logic - every output keeps its uploaded filename.
  - No ZIP packaging - each processed image gets its own download button.
  - No "Renaming and IS creation" workflow.

Requirements: streamlit, Pillow, rembg, onnxruntime
"""
import gc
import io
import re
import time
import importlib.util
from collections import Counter
from typing import Optional, Tuple

import streamlit as st
from PIL import Image, ImageChops, ImageOps, UnidentifiedImageError

# Don't import rembg/onnxruntime at module load time - they're ~200MB of ML
# libraries. find_spec() just checks the package is installed, it doesn't load it.
REMBG_AVAILABLE = importlib.util.find_spec("rembg") is not None


@st.cache_resource(show_spinner="Loading background-removal model (first time only)...")
def get_rembg_session():
    from rembg import new_session
    return new_session("u2netp")


def remove_background(img: Image.Image) -> Image.Image:
    from rembg import remove as rembg_remove
    session = get_rembg_session()
    result = rembg_remove(img.convert("RGBA"), session=session)
    return result if isinstance(result, Image.Image) else Image.open(io.BytesIO(result)).convert("RGBA")


st.set_page_config(page_title="Creative Editing - Unified", page_icon="🎨", layout="centered")

NAVY, BORDER = "#0B2E59", "#D9E2EC"
st.markdown(
    f"""
    <style>
    .stApp {{ background: white; }}
    h1, h2, h3 {{ color: {NAVY}; }}
    .small-note {{ color: #6b7280; font-size: 0.85rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# name -> (recommended_w, recommended_h, min, max, aspect_ratio)
MARKETPLACE_PRESETS = {
    "Custom": (800, 800, "—", "—", "—"),
    "Allegro PL — 2200 x 2200": (2200, 2200, "500 x 500", "2560 x 2560", "1:1"),
    "Allegro One PL — 2200 x 2200": (2200, 2200, "1000 x 1000", "2560 x 2560", "1:1"),
    "Best Buy US — 2000 x 2000": (2000, 2000, "2000 x 2000", "—", "1:1"),
    "Best Buy CA — 2000 x 2000": (2000, 2000, "2000 x 2000", "—", "1:1"),
    "Bol NL — 2400 x 2400": (2400, 2400, "500 x 500", "6000 x 6000", "1:1"),
    "Bol BE — 2400 x 2400": (2400, 2400, "500 x 500", "6000 x 6000", "1:1"),
    "eBay US — 1600 x 1600": (1600, 1600, "500 x 500", "9000 x 9000", "1:1"),
    "eBay DE — 1600 x 1600": (1600, 1600, "500 x 500", "9000 x 9000", "1:1"),
    "eBay UK — 1600 x 1600": (1600, 1600, "500 x 500", "9000 x 9000", "1:1"),
    "Kohl's US — 1000 x 1000": (1000, 1000, "1000 x 1000", "—", "1:1"),
    "Lowes US — 1000 x 1000": (1000, 1000, "1000 x 1000", "—", "1:1"),
    "Macy's US — 1000 x 1000": (1000, 1000, "1000 x 1000", "—", "1:1"),
    "MediaMarkt DE — 1200 x 1200": (1200, 1200, "1000 x 1000", "—", "1:1"),
    "Mercado Libre US — 1600 x 1600": (1600, 1600, "500 x 500", "2500 x 2500", "1:1"),
    "Nordstrom US — 2600 x 4000": (2600, 4000, "1300 x 2000", "—", "2:3"),
    "Octopia FR — 500 x 500": (500, 500, "1000 x 1000", "2500 x 2500", "1:1"),
    "OTTO DE — 960 x 480": (960, 480, "—", "—", "2:1"),
    "Target US — 2400 x 2400": (2400, 2400, "1200 x 1200", "5000 x 5000", "1:1"),
    "Tesco UK — 2400 x 2400": (2400, 2400, "1000 x 1000", "—", "1:1"),
    "Tik Tok US — 1000 x 1000": (1000, 1000, "600 x 600", "3000 x 3000", "1:1"),
    "Tik Tok UK — 1000 x 1000": (1000, 1000, "600 x 600", "3000 x 3000", "1:1"),
    "Walmart US — 2200 x 2200": (2200, 2200, "1500 x 1500", "5000 x 5000", "1:1"),
    "Walmart CA — 2200 x 2200": (2200, 2200, "1500 x 1500", "5000 x 5000", "1:1"),
    "Zalando DE — 2000 x 2000": (2000, 2000, "800 x 1200", "5000 x 5000", "2:3"),
}
PRESETS = {name: (info[0], info[1]) for name, info in MARKETPLACE_PRESETS.items()}
EXT_MAP = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp", "GIF": "gif", "BMP": "bmp", "TIFF": "tif"}


# ---------- small shared helpers ----------

def safe_filename(name: str, fallback: str = "image") -> str:
    name = (name or "").strip() or fallback
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", "_", name).strip("._ ")
    return name or fallback


def unique_name(base_name: str, counter: Counter) -> str:
    """Only renames on an actual collision - otherwise the given name is kept untouched."""
    base_name = safe_filename(base_name, "image")
    stem, ext = (base_name.rsplit(".", 1) + [""])[:2] if "." in base_name else (base_name, "")
    ext = f".{ext}" if ext else ""
    key = base_name.lower()
    counter[key] += 1
    return base_name if counter[key] == 1 else f"{stem}_{counter[key]}{ext}"


def has_alpha(img: Image.Image) -> bool:
    return img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)


def flatten_to_background(img: Image.Image, bg_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Paste onto a solid background wherever the image is transparent.
    Used both for the 'Add white background' option and for JPEG export."""
    if has_alpha(img):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, bg_color)
        bg.paste(rgba, mask=rgba.split()[3])
        return bg
    return img.convert("RGB")


def _scale_to_fit(img: Image.Image, target_w: int, target_h: int) -> Tuple[int, int]:
    scale = min(target_w / img.width, target_h / img.height)
    return max(1, round(img.width * scale)), max(1, round(img.height * scale))


# ---------- background / padding operations ----------

def remove_padding(img: Image.Image, tolerance: int = 6) -> Image.Image:
    """Trims only obvious empty space: fully transparent, or a clearly white/off-white
    border. Never resizes or intentionally cuts the product."""
    if has_alpha(img):
        rgba = img.convert("RGBA")
        alpha_bbox = rgba.split()[3].getbbox()
        return rgba.crop(alpha_bbox) if alpha_bbox else rgba

    rgb = img.convert("RGB")
    w, h = rgb.size
    step_x, step_y = max(1, w // 30), max(1, h // 30)
    border_pixels = (
        [rgb.getpixel((x, 0)) for x in range(0, w, step_x)]
        + [rgb.getpixel((x, h - 1)) for x in range(0, w, step_x)]
        + [rgb.getpixel((0, y)) for y in range(0, h, step_y)]
        + [rgb.getpixel((w - 1, y)) for y in range(0, h, step_y)]
    )
    white_like = sum(1 for r, g, b in border_pixels if r >= 245 and g >= 245 and b >= 245)
    if white_like / max(1, len(border_pixels)) < 0.85:
        return img  # border isn't clearly white - don't risk cutting the product

    white_bg = Image.new("RGB", rgb.size, (255, 255, 255))
    diff = ImageChops.difference(rgb, white_bg).convert("L")
    mask = diff.point(lambda px: 255 if px > tolerance else 0)
    bbox = mask.getbbox()
    return img.crop(bbox) if bbox else img


# ---------- resize operations ----------

def resize_fit(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    new_w, new_h = _scale_to_fit(img, target_w, target_h)
    return img.resize((new_w, new_h), Image.LANCZOS)


def resize_exact_canvas(img: Image.Image, target_w: int, target_h: int, canvas_mode: str) -> Image.Image:
    new_w, new_h = _scale_to_fit(img, target_w, target_h)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    ox, oy = (target_w - new_w) // 2, (target_h - new_h) // 2

    transparent = canvas_mode == "Transparent padding"
    canvas = Image.new("RGBA" if transparent else "RGB", (target_w, target_h),
                        (255, 255, 255, 0) if transparent else (255, 255, 255))
    if resized.mode == "RGBA":
        canvas.paste(resized, (ox, oy), resized.split()[3])
    else:
        canvas.paste(resized.convert(canvas.mode), (ox, oy))
    return canvas


def resize_exact_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    scale = max(target_w / img.width, target_h / img.height)
    new_w, new_h = max(1, round(img.width * scale)), max(1, round(img.height * scale))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    left, top = max(0, (new_w - target_w) // 2), max(0, (new_h - target_h) // 2)
    return resized.crop((left, top, left + target_w, top + target_h))


RESIZE_DISPATCH = {
    "By Width only": lambda img, w, h, canvas_mode: img.resize(
        (w, max(1, round(img.height * w / img.width))), Image.LANCZOS),
    "By Height only": lambda img, w, h, canvas_mode: img.resize(
        (max(1, round(img.width * h / img.height)), h), Image.LANCZOS),
    "Exact size - full image with padding": lambda img, w, h, canvas_mode: resize_exact_canvas(img, w, h, canvas_mode),
    "Exact size - fill frame / crop edges": lambda img, w, h, canvas_mode: resize_exact_crop(img, w, h),
    "Resize image only - dimensions may differ": lambda img, w, h, canvas_mode: resize_fit(img, w, h),
}


# ---------- main processing ----------

def process_image(
    raw_bytes: bytes,
    filename: str,
    resize_mode: str,
    target_w: int,
    target_h: int,
    output_format: Optional[str],
    quality: int,
    output_dpi: int,
    bg_mode: str,
    padding_mode: str,
    canvas_mode: str,
) -> Tuple[bytes, str, int, int, Optional[Image.Image]]:
    with Image.open(io.BytesIO(raw_bytes)) as opened:
        ImageOps.exif_transpose(opened, in_place=True)
        original_format = opened.format or "PNG"
        img = opened.copy()

    if padding_mode == "Remove padding":
        img = remove_padding(img)

    if bg_mode == "Remove background":
        if not REMBG_AVAILABLE:
            raise RuntimeError("Background removal is selected but rembg/onnxruntime is not installed.")
        img = remove_background(img)
    elif bg_mode == "Add white background":
        img = flatten_to_background(img, (255, 255, 255))

    img = RESIZE_DISPATCH[resize_mode](img, target_w, target_h, canvas_mode)

    save_fmt = output_format or original_format
    if save_fmt not in ("PNG", "JPEG", "WEBP", "GIF", "BMP", "TIFF"):
        save_fmt = "PNG"
    if save_fmt == "JPEG":
        img = flatten_to_background(img, (255, 255, 255))
    elif save_fmt in ("PNG", "WEBP") and img.mode not in ("RGBA", "RGB", "L", "LA"):
        img = img.convert("RGBA" if has_alpha(img) else "RGB")

    buffer = io.BytesIO()
    save_kwargs = {
        "JPEG": {"quality": quality, "subsampling": 0, "optimize": True, "progressive": True, "dpi": (output_dpi, output_dpi)},
        "WEBP": {"quality": quality, "method": 6},
        "PNG": {"dpi": (output_dpi, output_dpi), "compress_level": 3},
    }.get(save_fmt, {})
    img.save(buffer, format=save_fmt, **save_kwargs)
    buffer.seek(0)
    output_data = buffer.read()

    stem = safe_filename(filename.rsplit(".", 1)[0] if "." in filename else filename, "image")
    output_name = f"{stem}.{EXT_MAP.get(save_fmt, save_fmt.lower())}"
    preview_img = img.copy() if img.width * img.height <= 16_000_000 else None
    width, height = img.size
    img.close()
    buffer.close()
    return output_data, output_name, width, height, preview_img


# ================= UI =================
st.title("Creative Editing")
st.caption("Upload one image or several. Keeps the original look unless you choose otherwise.")

st.divider()
st.subheader("Give me images")
uploaded_images = st.file_uploader(
    "Upload image files",
    type=["png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff"],
    accept_multiple_files=True,
)
if uploaded_images:
    st.success(f"{len(uploaded_images)} uploaded image(s) ready - original filenames will be kept.")
else:
    st.info("Upload one or more images to start.")
    st.stop()

st.divider()
st.subheader("Background")
bg_mode = st.radio("Choose one", ["Keep original", "Remove background", "Add white background"], horizontal=True, index=0)
if bg_mode == "Remove background" and not REMBG_AVAILABLE:
    st.warning("Background removal needs rembg and onnxruntime in requirements.txt.")

st.divider()
st.subheader("Padding")
padding_mode = st.radio("Padding option", ["Keep padding", "Remove padding"], horizontal=True, index=0)
st.caption("Keep padding preserves the image exactly. Remove padding only trims obvious empty white/transparent space.")

st.divider()
st.subheader("Dimensions")
preset = st.selectbox("Choose marketplace recommended size", list(PRESETS.keys()))
default_w, default_h = PRESETS[preset]
if preset != "Custom":
    rec_w, rec_h, min_dim, max_dim, aspect_ratio = MARKETPLACE_PRESETS[preset]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recommended", f"{rec_w} x {rec_h}")
    m2.metric("Minimum", min_dim)
    m3.metric("Maximum", max_dim)
    m4.metric("Ratio", aspect_ratio)
st.caption("To get exact marketplace dimensions the app uses padding or crop. Stretching is never used.")

resize_choice = st.radio(
    "Resize mode",
    ["By Width only", "By Height only", "Exact W x H"],
    horizontal=True,
    help="By Width/Height keeps the full image ratio. Exact W x H can guarantee the selected dimensions.",
)

resize_mode = resize_choice
canvas_mode = "White padding"
if resize_choice == "Exact W x H":
    resize_mode = st.radio(
        "Exact size behaviour",
        ["Exact size - full image with padding", "Exact size - fill frame / crop edges", "Resize image only - dimensions may differ"],
        index=0,
        help="Padding keeps the full image visible. Crop fills the frame but may cut edges.",
    )
    if resize_mode == "Exact size - full image with padding":
        canvas_mode = st.radio("Padding background", ["White padding", "Transparent padding"], horizontal=True, index=0)

col_w, col_h = st.columns(2)
target_w, target_h = default_w, default_h
if resize_choice in ("By Width only", "Exact W x H"):
    target_w = int(col_w.number_input("Width (px)", min_value=1, value=int(default_w), step=1))
if resize_choice in ("By Height only", "Exact W x H"):
    target_h = int(col_h.number_input("Height (px)", min_value=1, value=int(default_h), step=1))
output_dpi = int(st.number_input("DPI", min_value=50, max_value=1200, value=300, step=1))

st.divider()
st.subheader("Output Format")
chosen_format = st.selectbox("Output format", ["Keep original format", "PNG", "JPEG", "WEBP"], index=0)
output_format = None if chosen_format == "Keep original format" else chosen_format
quality = 98
if output_format in ("JPEG", "WEBP"):
    quality = st.slider(f"{output_format.title()} quality", 80, 100, 98, 1)
else:
    st.caption("Keeping original format avoids unnecessary conversion or compression.")
if resize_mode == "Exact size - full image with padding" and canvas_mode == "Transparent padding" and output_format == "JPEG":
    st.warning("JPEG cannot keep transparent padding. Choose PNG/WebP or use White padding.")

st.divider()
total_count = len(uploaded_images)
st.info(f"Ready to process {total_count} image(s).")

if st.button("Process images", type="primary", use_container_width=True):
    start_time = time.time()
    progress = st.progress(0, text="Starting...")
    status_box = st.empty()
    successful, errors = [], []
    name_counter: Counter = Counter()

    for index, file in enumerate(uploaded_images, start=1):
        filename = file.name
        try:
            status_box.info(f"Processing image {index} of {total_count}: {filename}")
            file.seek(0)
            raw_bytes = file.read()
            output_data, out_name, w, h, preview_img = process_image(
                raw_bytes=raw_bytes,
                filename=filename,
                resize_mode=resize_mode,
                target_w=target_w,
                target_h=target_h,
                output_format=output_format,
                quality=quality,
                output_dpi=output_dpi,
                bg_mode=bg_mode,
                padding_mode=padding_mode,
                canvas_mode=canvas_mode,
            )
            out_name = unique_name(out_name, name_counter)
            successful.append({"name": out_name, "data": output_data, "w": w, "h": h})
        except UnidentifiedImageError:
            errors.append({"File": filename, "Error": "File could not be opened as an image"})
        except Exception as exc:
            errors.append({"File": filename, "Error": str(exc)})

        progress.progress(index / total_count, text=f"Processed {index} of {total_count}")
        if index % 25 == 0:
            gc.collect()

    elapsed = time.time() - start_time
    progress.empty()
    status_box.empty()

    st.session_state["result"] = {
        "successful": successful,
        "errors": errors,
        "elapsed": elapsed,
    }

result = st.session_state.get("result")
if result:
    st.success(f"Done. {len(result['successful'])} image(s) processed, {len(result['errors'])} failed.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Successful", len(result["successful"]))
    c2.metric("Failed", len(result["errors"]))
    c3.metric("Time", f"{result['elapsed']:.1f}s")

    if result["successful"]:
        st.subheader("Results")
        for item in result["successful"]:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(item["data"], caption=f"{item['name']} ({item['w']}x{item['h']})", use_container_width=True)
            with col2:
                st.download_button(
                    f"Download {item['name']}",
                    data=item["data"],
                    file_name=item["name"],
                    mime="application/octet-stream",
                    key=f"dl_{item['name']}",
                )

    if result["errors"]:
        with st.expander("View errors"):
            for e in result["errors"]:
                st.write(f"**{e['File']}**: {e['Error']}")

    if st.button("Clear results", use_container_width=True):
        del st.session_state["result"]
        st.rerun()

st.divider()
st.caption("Default behavior keeps the original image look. No shadow is added. Cropping happens only if Remove padding is selected. Output filenames match the name you uploaded.")
