"""
Creative Editing - Unified
Keeps the original image look by default. No forced white background,
no shadow, no cropping unless the user chooses Remove padding.

Simplified from the original "Creative Editing for Atlas" tool:
  - No custom filename/renaming logic - every output keeps its uploaded filename.
  - No ZIP packaging - each processed image gets its own download button.

Requirements: streamlit, Pillow, rembg, onnxruntime
"""
import io
import time
import importlib.util

import streamlit as st
from PIL import Image, ImageOps

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


def add_white_background(img: Image.Image) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.paste(img, mask=img)
    return bg.convert("RGB")


def trim_padding(img: Image.Image) -> Image.Image:
    if img.mode == "RGBA":
        bbox = img.getbbox()
    else:
        gray = img.convert("L")
        diff = ImageOps.invert(gray).point(lambda p: 255 if p > 10 else 0)
        bbox = diff.getbbox()
    return img.crop(bbox) if bbox else img


DIM_PRESETS = {
    "Original": None,
    "1000x1000 (Amazon/Etsy)": (1000, 1000),
    "1024x1024 (Shopify)": (1024, 1024),
    "2000x2000 (eBay)": (2000, 2000),
}


def resize_to(img: Image.Image, target) -> Image.Image:
    if target is None:
        return img
    img_copy = img.copy()
    img_copy.thumbnail(target, Image.LANCZOS)
    canvas_mode = "RGBA" if img_copy.mode == "RGBA" else "RGB"
    fill = (255, 255, 255, 0) if canvas_mode == "RGBA" else (255, 255, 255)
    canvas = Image.new(canvas_mode, target, fill)
    offset = ((target[0] - img_copy.width) // 2, (target[1] - img_copy.height) // 2)
    canvas.paste(img_copy, offset, img_copy if canvas_mode == "RGBA" else None)
    return canvas


def process_image(img: Image.Image, bg_option: str, padding_option: str, dim_target) -> Image.Image:
    result = img.convert("RGBA")

    if bg_option == "Remove background":
        if not REMBG_AVAILABLE:
            raise RuntimeError("Background removal is selected but rembg/onnxruntime is not installed.")
        result = remove_background(result)
    elif bg_option == "Add white background":
        result = add_white_background(result)

    if padding_option == "Remove padding":
        result = trim_padding(result)

    if dim_target is not None:
        result = resize_to(result, dim_target)

    return result


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Creative Editing - Unified", page_icon="🎨", layout="centered")

st.title("Creative Editing")
st.caption("Upload one image or several. Keeps the original look unless you choose otherwise.")

st.header("Background")
bg_option = st.radio(
    "Choose one",
    ["Keep original", "Remove background", "Add white background"],
    horizontal=True,
)
if bg_option == "Remove background" and not REMBG_AVAILABLE:
    st.warning("Background removal needs rembg and onnxruntime in requirements.txt.")

st.header("Padding")
padding_option = st.radio(
    "Padding option",
    ["Keep padding", "Remove padding"],
    horizontal=True,
)
st.caption(
    "Keep padding preserves the image exactly. "
    "Remove padding only trims obvious empty white/transparent space."
)

st.header("Dimensions")
size_choice = st.selectbox("Choose marketplace recommended size", list(DIM_PRESETS.keys()))
dim_target = DIM_PRESETS[size_choice]

st.header("Upload")
uploaded_files = st.file_uploader(
    "Upload one or more images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if uploaded_files and st.button(f"Process {len(uploaded_files)} image(s)"):
    start = time.time()
    successful, failed = [], []

    progress = st.progress(0, text="Starting...")
    for i, file in enumerate(uploaded_files):
        try:
            img = Image.open(file)
            result = process_image(img, bg_option, padding_option, dim_target)
            successful.append((file.name, result))
        except Exception as e:
            failed.append((file.name, str(e)))
        progress.progress((i + 1) / len(uploaded_files), text=f"Processed {i + 1}/{len(uploaded_files)}")

    progress.empty()
    elapsed = time.time() - start

    st.success(f"Done. {len(successful)} image(s) processed, {len(failed)} failed.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Successful", len(successful))
    c2.metric("Failed", len(failed))
    c3.metric("Time", f"{elapsed:.1f}s")

    if successful:
        st.subheader("Results")
        for name, img in successful:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(img, caption=name, use_column_width=True)
            with col2:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                st.download_button(
                    f"Download {name}",
                    data=buf.getvalue(),
                    file_name=name.rsplit(".", 1)[0] + ".png",
                    mime="image/png",
                    key=f"dl_{name}",
                )

    if failed:
        with st.expander("View errors"):
            for name, err in failed:
                st.write(f"**{name}**: {err}")
