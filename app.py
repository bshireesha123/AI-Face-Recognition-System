import streamlit as st
import face_recognition
import cv2
import numpy as np
import os

# Page settings
st.set_page_config(
    page_title="AI Face Recognition System",
    page_icon="🤖"
)

st.title("🤖 AI-Based Face Recognition System")
st.write("Capture your face and check whether you are a registered person.")

# Folder containing registered faces
KNOWN_FACES_DIR = "known_faces"

# Create folder if it doesn't exist
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

known_face_encodings = []
known_face_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):

        image_path = os.path.join(KNOWN_FACES_DIR, filename)

        image = face_recognition.load_image_file(image_path)

        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:

            known_face_encodings.append(encodings[0])

            name = os.path.splitext(filename)[0]
            known_face_names.append(name)

# Display registered people
st.subheader("👤 Registered People")

if known_face_names:

    for name in known_face_names:
        st.write("✅", name)

else:

    st.warning("No registered faces found.")

# Camera
st.subheader("📷 Camera")

camera_image = st.camera_input("Take a picture")

# Recognition
if camera_image is not None:

    image_bytes = camera_image.getvalue()

    image_array = np.frombuffer(image_bytes, np.uint8)

    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)

    # Encode detected faces
    face_encodings = face_recognition.face_encodings(
        rgb_frame,
        face_locations
    )

    st.subheader("🔍 Recognition Result")

    if len(face_encodings) == 0:

        st.error("❌ No face detected.")

    else:

        for face_encoding in face_encodings:

            matches = face_recognition.compare_faces(
                known_face_encodings,
                face_encoding,
                tolerance=0.6
            )

            face_distances = face_recognition.face_distance(
                known_face_encodings,
                face_encoding
            )

            name = "Unknown Person"

            if len(face_distances) > 0:

                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:

                    name = known_face_names[best_match_index]

            if name != "Unknown Person":

                st.success(f"✅ Recognized: {name}")

            else:

                st.error("❌ Unknown Person")

    # Display captured image
    st.image(
        rgb_frame,
        caption="Captured Image",
        use_container_width=True
    )