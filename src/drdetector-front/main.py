import streamlit as st
import requests
from urllib.parse import urljoin
from PIL import Image

# TODO: Change this into an env variable
BACKEND_URL = "http://localhost:8000/"

st.title("Diabetic Retinopathy Detector")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

options = ["Option 1", "Option 2", "Option 3", "Option 4"]

# Display the selectbox
selected_option = st.selectbox("Choose an option:", options)

# Display the selected option
st.write("You selected:", selected_option)

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Step 2: Specify the API endpoint
    #api_url = st.text_input("Enter the API URL", "http://example.com/api/upload")
    api_url = urljoin(BACKEND_URL, "predict")

    # Step 3: Send the image as a POST request
    if st.button("Send Image"):
        try:
            # Sending POST request
            response = requests.post(
                api_url,
                files={"file": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}
            )
            response_dict = response.json()
            # Display the response
            st.write("Response Status Code:", response.status_code)
            st.write("Image Class:", response_dict["image_class"])
        except Exception as e:
            st.error(f"An error occurred: {e}")

print("Uploaded image Type:", uploaded_image.type)
print("Uploaded image Type:", type(uploaded_image))
