import io
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


def load_image(uploaded_file):
    """
    Load and display an image from the uploaded file.

    Parameters:
    uploaded_file (UploadedFile): The uploaded image file.

    Returns:
    Image: The loaded image in RGB format.
    """
    if uploaded_file is None:
        return None
    else:
        image_data = uploaded_file.getvalue()
        st.image(image_data, width=200)
        st.write("Image Input : ", uploaded_file.name)
        return Image.open(io.BytesIO(image_data)).convert('RGB')


# Initialize the BLIP processor with the pretrained model
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-large"
    )


@st.cache_resource
def load_model():
    """
    Load the BLIP model for conditional generation.

    Returns:
    BlipForConditionalGeneration: The loaded BLIP model.
    """
    return BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
        )


# Load the model using the cached function
model = load_model()

# Streamlit application title and team member information
st.title("Project - Image to Text")
st.write("""
         #### TEAM MEMBER
         - Рахарди Сандикха РИМ-130908
         - Мухин Виктор Александрович РИМ-130908
         """)

st.write("""#### Our Project""")

# File uploader for image input
uploadedFile = st.file_uploader('Upload image here')
raw_image = load_image(uploadedFile)

# Submit button for generating image caption
result = st.button('Submit')

if result:
    # Generate caption for the uploaded image
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=1000)
    text_output = processor.decode(out[0], skip_special_tokens=True)
    st.write("=========================================")
    st.write("Output : ", str(text_output).capitalize())


# Documentation
"""
This script uses Streamlit to create a web application that generates textual
descriptions from uploaded images using the BLIP
(Bootstrapping Language-Image Pre-training) model.

#### Functions

- **`load_image(uploaded_file)`**: This function loads an image file
uploaded by the user, displays it, and returns the image in RGB format,
or None if no file is uploaded.

#### Cached Functions

- **`load_model()`**: This function loads the BLIP model for conditional
generation using a cached resource to avoid reloading the model
on every execution.

#### Streamlit Components

- **Title and Team Information**:
Displays the project title and team member information.
- **Image Uploader**:
Allows users to upload an image file.
- **Submit Button**:
When clicked, processes the uploaded image to generate
a caption using the BLIP model.

#### Workflow

1. **Upload Image**:
The user uploads an image using the file uploader.
2. **Display Image**:
The uploaded image is displayed on the web application.
3. **Generate Caption**:
Upon clicking the submit button, the image is processed,
and a caption is generated and displayed.
If no image is uploaded, a prompt asks the user to upload an image.
"""
