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
