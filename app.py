import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw
import io
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Lite Face Recognition", layout="centered")
st.title("🧠 Lite Face Recognition App (No dlib / No face_recognition)")

# Initialize mediapipe face detector
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Initialize mediapipe face mesh (for embeddings)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Initialize session storage
if "known_faces" not in st.session_state:
    st.session_state.known_faces = []  # (name, embedding)
    st.session_state.face_thumbnails = []  # (name, image)

# Helper: Detect face and get cropped region
def detect_and_crop_faces(image_np):
    results = face_detection.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    face_crops = []
    boxes = []
    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            h, w, _ = image_np.shape
            x1 = max(0, int(bbox.xmin * w))
            y1 = max(0, int(bbox.ymin * h))
            x2 = min(w, x1 + int(bbox.width * w))
            y2 = min(h, y1 + int(bbox.height * h))
            face_crops.append(image_np[y1:y2, x1:x2])
            boxes.append((x1, y1, x2, y2))
    return face_crops, boxes

# Helper: Get facial landmarks as embeddings
def get_face_embedding(image_np):
    results = face_mesh.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    return None

st.header("📌 Step 1: Register a Known Face")
with st.form("register", clear_on_submit=True):
    reg_img = st.file_uploader("Upload face image", type=["jpg", "jpeg", "png"], key="reg")
    reg_name = st.text_input("Name", key="name")
    submit = st.form_submit_button("Register")

if submit and reg_img and reg_name:
    image_np = np.array(Image.open(reg_img))
    crops, _ = detect_and_crop_faces(image_np)
    if crops:
        emb = get_face_embedding(crops[0])
        if emb is not None:
            st.session_state.known_faces.append((reg_name, emb))
            thumb = Image.fromarray(crops[0])
            thumb.thumbnail((100, 100))
            st.session_state.face_thumbnails.append((reg_name, thumb))
            st.success(f"✅ Registered {reg_name}")
        else:
            st.warning("No landmarks found. Try another image.")
    else:
        st.warning("No face detected. Try another image.")

if st.session_state.face_thumbnails:
    st.subheader("🖼️ Registered Gallery")
    cols = st.columns(len(st.session_state.face_thumbnails))
    for idx, (name, thumb) in enumerate(st.session_state.face_thumbnails):
        with cols[idx]:
            st.image(thumb, caption=name)

st.header("📷 Step 2: Recognize Faces in Uploaded Image")
query_img = st.file_uploader("Upload query image", type=["jpg", "jpeg", "png"], key="query")

if query_img and st.session_state.known_faces:
    image_np = np.array(Image.open(query_img))
    face_crops, boxes = detect_and_crop_faces(image_np)
    pil_img = Image.fromarray(image_np)
    draw = ImageDraw.Draw(pil_img)

    for crop, (x1, y1, x2, y2) in zip(face_crops, boxes):
        query_emb = get_face_embedding(crop)
        if query_emb is not None:
            similarities = [cosine_similarity([query_emb], [known[1]])[0][0] for known in st.session_state.known_faces]
            best_idx = int(np.argmax(similarities))
            best_score = similarities[best_idx]
            name = st.session_state.known_faces[best_idx][0] if best_score > 0.8 else "Unknown"
        else:
            name = "Unknown"

        color = "blue" if name != "Unknown" else "red"
        draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=3)
        draw.text((x1, y2 + 5), name, fill=color)

    st.image(pil_img, caption="Recognition Result", use_column_width=True)
