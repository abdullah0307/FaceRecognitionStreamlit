import face_recognition
import numpy as np
import streamlit as st
from PIL import Image

st.title("👤 Face Recognition App")
st.markdown("Upload a test face image and match it with the uploaded reference images.")

# Step 1: Upload reference face images
st.subheader("Step 1: Upload Reference Face Images")
reference_files = st.file_uploader("Upload one or more reference images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Display reference images in a matrix
if reference_files:
    st.markdown("### Uploaded Reference Images")
    cols = st.columns(4)
    for idx, ref_file in enumerate(reference_files):
        with cols[idx % 4]:
            img = Image.open(ref_file)
            st.image(img, caption=ref_file.name, use_container_width =True)

# Step 2: Upload test image
st.subheader("Step 2: Upload a Test Image")
test_file = st.file_uploader("Upload a test image", type=["jpg", "jpeg", "png"])

if test_file:
    st.markdown("### Uploaded Test Image")
    test_img = Image.open(test_file).convert("RGB")
    st.image(test_img, caption="Test Image", use_container_width=True)

# Matching Logic
if reference_files and test_file:
    st.subheader("Step 3: Matching Result")

    test_np = np.array(test_img)
    test_faces = face_recognition.face_encodings(test_np)

    if not test_faces:
        st.error("❌ No face detected in the test image.")
    else:
        test_encoding = test_faces[0]

        best_score = 1.0
        best_match_name = None
        best_match_image = None

        for ref_file in reference_files:
            ref_img = Image.open(ref_file).convert("RGB")
            ref_np = np.array(ref_img)

            ref_faces = face_recognition.face_encodings(ref_np)
            if not ref_faces:
                continue

            ref_encoding = ref_faces[0]
            distance = face_recognition.face_distance([ref_encoding], test_encoding)[0]

            if distance < best_score:
                best_score = distance
                best_match_name = ref_file.name
                best_match_image = ref_img

        if best_score < 0.4:
            st.success(f"✅ Match Found: `{best_match_name}` (Distance: {best_score:.2f})")
            st.image(best_match_image, caption=f"Matched Image: {best_match_name}", use_container_width=True)
        else:
            st.warning("❌ No good match found (all distances > 0.6).")
