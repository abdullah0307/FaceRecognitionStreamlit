import base64
import streamlit as st
import face_recognition
import numpy as np
from PIL import Image, ImageDraw
import io

st.set_page_config(page_title="Face Recognition App", layout="centered", initial_sidebar_state="collapsed")
st.title("🧠 Face Recognition App with Named Registration and Gallery")

hide_sidebar = """
    <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        [data-testid="stSidebarNav"] {
            display: none;
        }
        [data-testid="collapsedControl"] {
            display: none;
        }
    </style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)


# Session state initialization
if 'known_faces' not in st.session_state:
    st.session_state.known_faces = []
    st.session_state.face_names = []
    st.session_state.face_thumbnails = []

# Draw bounding boxes
def draw_boxes_on_image(image_array, face_locations, name=None):
    pil_img = Image.fromarray(image_array)
    draw = ImageDraw.Draw(pil_img)
    for top, right, bottom, left in face_locations:
        draw.rectangle(((left, top), (right, bottom)), outline="green", width=3)
        if name:
            draw.text((left + 6, bottom + 5), name, fill="green")
    return pil_img

st.header("Step 1: Register Known Faces")

# Name input first
with st.form("register_form", clear_on_submit=True):
    uploaded_file = st.file_uploader("Upload a known face image", type=["jpg", "jpeg", "png"], key="reg_img")
    name = st.text_input("Enter name for the face", key="reg_name")
    submitted = st.form_submit_button("Register Face")

if submitted and uploaded_file and name:
    image = face_recognition.load_image_file(uploaded_file)
    face_locations = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, face_locations)

    if encodings:
        st.session_state.known_faces.append(encodings[0])
        st.session_state.face_names.append(name)

        # Save thumbnail for gallery
        thumbnail = Image.fromarray(image)
        thumbnail.thumbnail((150, 150))
        st.session_state.face_thumbnails.append((name, thumbnail))

        boxed = draw_boxes_on_image(image, face_locations, name)
        st.image(boxed, caption=f"✅ Registered: {name}", use_container_width=True)
    else:
        st.warning("❌ No face found in the uploaded image.")



if st.session_state.face_thumbnails:
    st.subheader("🖼️ Registered Face Gallery")
    with st.container():
        st.markdown(
            """
            <div style="display: flex; overflow-x: auto; gap: 20px; padding-bottom: 10px;">""",
            unsafe_allow_html=True,
        )
        for face_name, thumb in st.session_state.face_thumbnails:
            bio = io.BytesIO()
            thumb.save(bio, format="PNG")
            b64 = base64.b64encode(bio.getvalue()).decode("utf-8")
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <img src="data:image/png;base64,{b64}" style="width: 100px; height: 100px; object-fit: cover; border-radius: 5px; border: 2px solid #ccc;">
                    <p style="font-size: 14px;">{face_name}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


# Query image
st.header("Step 2: Upload Query Image for Matching")
query_file = st.file_uploader("Upload a query image", type=["jpg", "jpeg", "png"], key="query_img")

if query_file and st.session_state.known_faces:
    query_img = face_recognition.load_image_file(query_file)
    face_locations = face_recognition.face_locations(query_img)
    query_encodings = face_recognition.face_encodings(query_img, face_locations)
    matched_name = ""

    if not query_encodings:
        st.error("❌ No face found in the query image.")
    else:
        match_found = False
        pil_img = Image.fromarray(query_img)
        draw = ImageDraw.Draw(pil_img)

        for encoding, (top, right, bottom, left) in zip(query_encodings, face_locations):
            matches = face_recognition.compare_faces(st.session_state.known_faces, encoding)
            distances = face_recognition.face_distance(st.session_state.known_faces, encoding)
            best_match_index = np.argmin(distances)

            if matches[best_match_index]:
                match_found = True
                matched_name = st.session_state.face_names[best_match_index]
                draw.rectangle(((left, top), (right, bottom)), outline="blue", width=3)
                draw.text((left + 6, bottom + 5), matched_name, fill="blue")
            else:
                draw.rectangle(((left, top), (right, bottom)), outline="red", width=3)
                draw.text((left + 6, bottom + 5), "Unknown", fill="red")

        if match_found:
            st.image(pil_img, caption="✅ Match Found", use_container_width=True)
            st.success(f"Match found with a registered face {matched_name}")
        else:
            st.image(pil_img, caption="❌ No Match Found", use_container_width=True)
            st.warning("No match found in registered faces.")
