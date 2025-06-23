import streamlit as st
from deepface import DeepFace
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Face Match", layout="wide")
st.title("🔍 Face Recognition App (No dlib)")

# Upload reference images
st.subheader("Step 1: Upload Reference Face Images")
reference_files = st.file_uploader("Upload one or more reference images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Display reference images
if reference_files:
    st.markdown("### Reference Image Gallery")
    cols = st.columns(4)
    for i, file in enumerate(reference_files):
        img = Image.open(file).convert("RGB")
        with cols[i % 4]:
            st.image(img, caption=file.name, use_column_width=True)

# Upload test image
st.subheader("Step 2: Upload a Test Image to Match")
test_file = st.file_uploader("Upload test face image", type=["jpg", "jpeg", "png"])

if test_file:
    test_img = Image.open(test_file).convert("RGB")
    st.image(test_img, caption="Test Image", use_column_width=True)

# Match faces using DeepFace
if test_file and reference_files:
    st.subheader("Step 3: Matching Result")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as test_temp:
        test_img.save(test_temp.name)
        test_path = test_temp.name

    best_match = None
    best_distance = float("inf")
    match_name = None

    for ref_file in reference_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as ref_temp:
            img = Image.open(ref_file).convert("RGB")
            img.save(ref_temp.name)
            ref_path = ref_temp.name

        try:
            result = DeepFace.verify(img1_path=test_path, img2_path=ref_path, enforce_detection=False)
            distance = result["distance"]
            if result["verified"] and distance < best_distance:
                best_match = img
                best_distance = distance
                match_name = ref_file.name
        except Exception as e:
            st.warning(f"Error processing {ref_file.name}: {e}")

    if best_match:
        st.success(f"✅ Match Found: `{match_name}` (Distance: {best_distance:.2f})")
        st.image(best_match, caption=f"Best Match: {match_name}", use_column_width=True)
    else:
        st.error("❌ No match found among reference images.")
