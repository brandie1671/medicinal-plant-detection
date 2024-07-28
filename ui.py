import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('model_sev.h5')

# Define class names
class_names = ['Arive-Dantu', 'Basale', 'Betel', 'Crape_Jasmine', 'Curry', 'Drumstick', 'Fenugreek', 'Guava', 
               'Hibiscus', 'Indian_Beech', 'Indian_Mustard', 'Jackfruit', 'Jamaica_Cherry-Gasagase', 'Jamun', 
               'Jasmine', 'Karanda', 'Lemon', 'Mango', 'Mexican_Mint', 'Mint', 'Neem', 'Oleander', 'Parijata', 
               'Peepal', 'Pomegranate', 'Rasna', 'Rose_apple', 'Roxburgh_fig', 'Sandalwood', 'Tulsi']

# Define medicinal properties for each class
medicinal_properties = {
    'Arive-Dantu': 'Rich in fiber, low in calories, used for weight loss, and has antioxidant properties.',
    'Basale': 'Anti-inflammatory and wound-healing properties.',
    'Betel': 'Used to improve mood, aid in digestion, and possess antimicrobial properties for oral health.',
    'Crape_Jasmine': 'Used in the treatment of liver diseases, abdominal pain, and to improve mood.',
    'Curry': 'Aids digestion, lowers blood cholesterol, promotes hair growth, and has anti-inflammatory properties.',
    'Drumstick': 'Boosts immunity, relieves joint pain, and aids in bone health.',
    'Fenugreek': 'Helps in metabolic conditions like diabetes, heartburn, and obesity prevention.',
    'Guava': 'Rich in Vitamin C and antioxidants, aids in preventing infections, and supports heart and kidney health.',
    'Hibiscus': 'Lowers blood pressure, relieves dry coughs, and may have anti-inflammatory effects.',
    'Indian_Beech': 'Used for skin disorders, wound healing, and has antimicrobial properties.',
    'Indian_Mustard': 'Relieves joint pain, inflammation, and is heart-healthy.',
    'Jackfruit': 'Rich in Vitamin A, promotes eye health, and has antioxidant properties.',
    'Jamaica_Cherry-Gasagase': 'Anti-diabetic, boosts immunity, and aids in digestion.',
    'Jamun': 'Treats common cold, cough, flu, and sore throat problems.',
    'Jasmine': 'Similar medicinal uses to Crape Jasmine.',
    'Karanda': 'Used for digestion problems, respiratory infections, and dermatitis.',
    'Lemon': 'Rich in Vitamin C, aids in digestion, and prevents kidney stones.',
    'Mango': 'Rich in vitamins and antioxidants, promotes heart health, and aids in digestion.',
    'Mexican_Mint': 'Treats respiratory illness, skincare, and may have anti-inflammatory effects.',
    'Mint': 'Relieves indigestion, upset stomach, and has nutritional benefits.',
    'Neem': 'Used for skin diseases, boosts immunity, and acts as an insect repellent.',
    'Oleander': 'Used for heart conditions, asthma, epilepsy, and leprosy (caution: can be poisonous).',
    'Parijata': 'Anti-inflammatory, antipyretic, and used for pain relief.',
    'Peepal': 'Improves complexion, strengthens blood capillaries, and has anti-inflammatory effects.',
    'Pomegranate': 'Rich in antioxidants, heart-healthy, and may protect against Alzheimer\'s.',
    'Rasna': 'Relieves bone and joint pain, treats respiratory issues, and aids wound healing.',
    'Rose_apple': 'Treats asthma, fever, inflammation, and has antimicrobial properties.',
    'Roxburgh_fig': 'Used in wound healing and treating diarrhea.',
    'Sandalwood': 'Treats cold, cough, bronchitis, fever, and may have other medicinal uses.',
    'Tulsi': 'Boosts immunity, treats fever, respiratory problems, and has skincare benefits.'
}

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (150, 150))
    image = image / 255.0
    return image

# Function to predict and display medicinal properties
def predict_and_display_properties(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(np.array([processed_image]))
    pred_label = np.argmax(prediction)
    class_prediction = class_names[pred_label]
    st.write("Detected Plant:", class_prediction)
    if class_prediction in medicinal_properties:
        st.write("Medicinal Properties:", medicinal_properties[class_prediction])
    else:
        st.write("Medicinal properties not found.")

# Streamlit App
st.title("Plant Recognition and Medicinal Properties")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    if st.button('Identify Plant and Display Medicinal Properties'):
        predict_and_display_properties(image)
