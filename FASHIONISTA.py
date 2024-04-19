import streamlit as st
import pandas as pd
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
import io
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import base64
import plotly.express as px

# Load data and models
embeddings = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('images.pkl', 'rb'))
myntra = pd.read_csv('myntra.csv', index_col=0)
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


st.markdown('<h1 style="font-family: audrey; font-size: 75px; font-weight: bold; text-shadow: 40px 40px 80px rgba(0, 0, 0, 1.11);">FASHIONISTA</h1>', unsafe_allow_html=True)




def feature_extraction(img_array,model):
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices




def display_images(indices, df):
    st.subheader("Similar Images:")
    for idx in indices[0]:
        image_links = df.iloc[idx][['img1', 'img2', 'img3', 'img4']].tolist()
        
        # Display the first image
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            image_expanded = st.image(image_links[0], caption=f"Image {idx+1}", use_column_width=True)
        
        # Calculate the total width of all images for setting the width of the box
        total_image_width = len(image_links) * (300 + 20)  # Width of image + margin
        
        # Create a horizontal rectangle box for additional information
        with st.container():
            st.markdown(
                f"""
                <style>
                .horizontal-box {{
                    background-color: black;
                    color: white;
                    padding: 10px;
                    border-radius: 5px;
                    display: flex;
                    flex-direction: row;
                    align-items: center;
                    width: min({total_image_width}px, 100%);  /* Set the width dynamically */
                    margin-top: 10px;  /* Add some margin from the images */
                }}
                </style>
                <div class="horizontal-box">
                    <p style="margin: 0;"> </p>
                    <div style="flex-grow: 1; padding-left: 10px;">
                        <p style="margin: 0;"><strong>Occasion:</strong> {df.iloc[idx]['occasion']}</p>
                        <p style="margin: 0;"><strong>Preferred Body Type:</strong> {df.iloc[idx]['bodyshape']}</p>
                        <p style="margin: 0;"><strong>Check Out:</strong> <a href="{df.iloc[idx]['link']}" target="_blank">{df.iloc[idx]['link']}</a></p>
                        <p style="margin: 0;"><strong>Outfit Inspo:</strong> {df.iloc[idx]['complete_the_look']}</p>
                    </div>
                </div>
                """
            , unsafe_allow_html=True)
        
        # Display the remaining images
        for image_link in image_links[1:]:
            col1, col2, col3, col4 = col2, col3, col4, st.columns(1)
            with col1:
                st.image(image_link, caption="", use_column_width=True)



df = px.data.iris()

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("image.jpeg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://images.unsplash.com/photo-1609709295948-17d77cb2a69b?q=80&w=1888&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: local;
}}

.title {{
    font-size: 48px; /* Adjust the font size for the title */
    font-weight: bold; /* Make the font bold */
    color: white; /* Set title color to white */
    text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.5); /* Add text shadow */
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


def home_page():
    # Set background color and padding
    background_style = """
        background-color: rgba(0, 0, 0, 0.5); /* Semi-transparent black background */
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px; /* Add margin at the bottom */
    """

    # Title and introductory text styling
    title_style = """
        color: #ffffff; /* White color */
        font-size: 28px; /* Font size */
        font-weight: bold; /* Bold font weight */
        margin-bottom: 20px; /* Add margin at the bottom */
    """


    background_style = "background-color: #000000; padding: 8px;"
    title_style = "color: #F5F5DC; font-size: 24px; font-weight: bold;"

    st.markdown(
        f"""
        <div style="{background_style}">
            <h1 style="{title_style}">Unlock Your Potential with Fashionista!</h1>
            <p style="color: #F0F0F0; font-size: 16px; margin-bottom: 20px;">Ever feel overwhelmed by clothing choices? Unsure what styles flatter your body shape? Or struggle to find similar products you love? We've all been there!<br><br>Introducing Fashionista! Here, you can explore two exciting features that take the guesswork out of dressing well:</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Add space between sections
    st.markdown("---")

# Features
    feature_style = """
        background-color: #333333;
        color: #ffffff;
        padding: 8px;
        border-radius: 10px;
        margin-bottom: 20px; /* Add margin at the bottom to separate features */
    """

# Body Shape Calculator
    st.markdown(
        f"""
        <div style="{feature_style}">
            <h2 style="margin-bottom: 10px;">Body Shape Calculator</h2>
            <p>Discover your unique body shape and understand clothing cuts that flatter you the most. No more endless browsing or trying on clothes that don't work for you.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Recommendation
    st.markdown(
        f"""
        <div style="{feature_style}">
            <h2 style="margin-bottom: 10px;">Recommendation</h2>
            <p>Upload an image of your favorite outfit and get a curated selection of similar products to explore! Find new pieces you'll love or discover similar styles from different brands.</p>
        </div>
        """,
        unsafe_allow_html=True
    )



def recommendation_page():
    st.markdown('<h1 style="text-align: left; font-weight: bold; color: white;">Similar Recommendations</h1>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        try:
            image_bytes = uploaded_file.getvalue()
            display_image = Image.open(io.BytesIO(image_bytes))
            
            # Resize the image to a larger size
            large_image = display_image.resize((800, 800))
            
            # Resize the large image to match the expected input shape of the model
            resized_image = large_image.resize((224, 224))
            
            st.image(large_image)
            
            img_array = np.array(resized_image)
            features = feature_extraction(img_array, model)
            indices = recommend(features, embeddings)
            display_images(indices, myntra)
        except Exception as e:
            st.error(f"Error occurred: {e}")    


def calculate_body_shape(bust, waist, hip):
    #st.title("Body Shape Calculator")
    #st.subheader("Enter your measurements to calculate your body shape and type.")
    body_shape = ""

    # Ensure measurements are valid floats
    try:
        bust = float(bust)
        waist = float(waist)
        hip = float(hip)
    except ValueError:
        return "Error: Please enter valid numbers for measurements."

    # Calculate body shape based on ratios
    if waist * 1.25 <= bust and waist <= hip:
        body_shape = "Hourglass"
    elif hip * 1.05 > bust:
        body_shape = "Pear"
    elif hip * 1.05 < bust:
        body_shape = "Apple"
    else:
        high = max(bust, waist, hip)
        low = min(bust, waist, hip)
        difference = high - low
        if difference <= 5:
            body_shape = "Rectangle"

    return body_shape

def calculate_body_type(bmi, body_shape):
    body_type = ""

    # Ensure BMI is a valid float
    try:
        bmi = float(bmi)
    except ValueError:
        return "Error: Please enter a valid number for BMI."

    if body_shape == "Error":
        return "Error: Body shape calculation failed."

    type_descriptor = ""
    if 1 <= bmi < 18:
        type_descriptor = "A"
    elif 18 <= bmi < 23:
        type_descriptor = "B"
    elif 23 <= bmi < 29:
        type_descriptor = "C"
    elif 29 <= bmi < 55:
        type_descriptor = "D"
    elif bmi >= 55:
        type_descriptor = "E"

    if type_descriptor == "A":
        body_type = "Skinny"
    elif type_descriptor == "B":
        body_type = "Petite"
    elif type_descriptor == "C" and body_shape != "Hourglass":
        body_type = "Average"
    elif type_descriptor == "C" and body_shape == "Hourglass":
        body_type = "Curvy"
    elif type_descriptor == "D" and body_shape == "Rectangle":
        body_type = "BBW"
    elif type_descriptor == "D" and (body_shape == "Hourglass" or body_shape == "Curvy"):
        body_type = "BBW - Curvy"
    elif type_descriptor == "D" and body_shape == "Pear":
        body_type = "BBW - Bottom Heavy"
    elif type_descriptor == "D" and body_shape == "Apple":
        body_type = "BBW - Top Heavy"
    elif type_descriptor == "E" and (body_shape == "Rectangle" or body_shape == "Hourglass"):
        body_type = "SSBBW"
    elif type_descriptor == "E" and body_shape == "Apple":
        body_type = "SSBBW - Top Heavy"
    elif type_descriptor == "E" and body_shape == "Pear":
        body_type = "SSBBW - Bottom Heavy"

    return body_type




def body_shape_page():
    # Define the CSS for the background image
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
        background-image: url("https://images.unsplash.com/photo-1584184924103-e310d9dc82fc?q=80&w=1887&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        padding: 20px;
        border-radius: 10px;
    }}
    </style>
    """

    # Inject the CSS for the background image
    st.markdown(page_bg_img, unsafe_allow_html=True)

    # Display the body shape calculator within the styled container
    st.title("Calculate Your Body Shape")

    bust = st.number_input("Bust (inches):", min_value=0.0, step=0.1, key="bust")
    waist = st.number_input("Waist (inches):", min_value=0.0, step=0.1, key="waist")
    hip = st.number_input("Hip (inches):", min_value=0.0, step=0.1, key="hip")
    height_ft = st.number_input("Height (feet):", min_value=0.0, key="height_ft")
    height_in = st.number_input("Height (inches):", min_value=0.0, max_value=11.9, step=0.1, key="height_in")
    
    if st.button("Calculate"):
        if bust != 0 and waist != 0 and hip != 0 and height_ft != 0 and height_in != 0:
            # Calculate BMI (Body Mass Index)
            height_in_total = height_ft * 12 + height_in
            bmi = round((bust * bust) / (height_in_total * height_in_total) * 703, 2)  # Conversion factor for inches

            # Calculate body shape
            body_shape = calculate_body_shape(bust, waist, hip)

            # Calculate body type
            body_type = calculate_body_type(bmi, body_shape)

            # Display results
            if body_shape != "Error" and body_type != "Error":
                st.success(f"Your Body Shape: {body_shape}")
                st.success(f"Your Body Type: {body_type}")
                #st.write("**Note:** These results are for informational purposes only and should not be used for medical diagnosis. Please consult with a healthcare professional for personalized advice.")
        else:
            st.warning("Please enter all measurements to calculate your body shape and type.")


def main():
    st.sidebar.title("")
    st.sidebar.markdown("<h2>Go to:</h2>", unsafe_allow_html=True)
    page = st.sidebar.radio("", ["Home", "Similiar Recommendations", "Calculate Your Body Shape"], index=0)

    if page == "Home":
        home_page()
    elif page == "Similiar Recommendations":
        recommendation_page()
    elif page == "Calculate Your Body Shape":
        body_shape_page()
 

if __name__ == "__main__":
    main()
