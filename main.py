import io
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


def load_image(uploaded_file):
    """
    This function loads and processes the uploaded image.

    Args:
        uploaded_file: The uploaded image file object.

    Returns:
        PIL Image object in RGB mode if successful, otherwise None.
    """
    if uploaded_file is None:
        return None
    else:
        image_data = uploaded_file.getvalue()
        st.image(image_data, width=400)
        st.write("Image Input : ", uploaded_file.name)
        return Image.open(io.BytesIO(image_data)).convert('RGB')


# Initialize the BLIP processor with the pretrained model
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-large"
    )


@st.cache_resource
def load_model():
    """
    Loads the BLIP image captioning model using caching for efficiency.

    Returns:
    BlipForConditionalGeneration: The loaded BLIP model.
    """
    return BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
        )


# Load the model using the cached function
model = load_model()

# --- Streamlit User Interface ---

# Streamlit application title and team member information
st.title("Project - Image to Text")
st.write("""
         #### TEAM MEMBER
         - Рахарди Сандикха РИМ-130908
         - Мухин Виктор Александрович РИМ-130908
         """)
st.write("""#### Our Project: Image Caption Generator""")

# File uploader and image loading
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]  # Specify allowed types
    )

raw_image = load_image(uploaded_file)


# Caption generation and display
if st.button("Generate Caption"):
    if raw_image:  # Check if an image is loaded before proceeding
        # Add a spinner for visual feedback
        with st.spinner("Generating caption..."):
            inputs = processor(raw_image, return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=1000)
            caption = processor.decode(out[0], skip_special_tokens=True)
            # Nicer formatting
            st.markdown(f"**A picture of** {caption.capitalize().lower()}")
    else:
        # Provide feedback if no image is uploaded
        st.warning("Please upload an image.")
