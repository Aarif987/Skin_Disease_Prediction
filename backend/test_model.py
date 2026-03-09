import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore') # Hides annoying sci-kit warnings

# --- 1. SETTINGS & PATHS ---
# Automatically find the correct paths based on where the script is run
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "backend", "model", "skin_disease_prediction.h5") # Using the BEST model
CSV_PATH = os.path.join(BASE_DIR, "backend", "dataset", "HAM10000_metadata.csv")

CLASSES = {
    0: 'Actinic keratoses', 1: 'Basal cell carcinoma', 2: 'Benign keratosis-like lesions',
    3: 'Dermatofibroma', 4: 'Melanoma', 5: 'Melanocytic nevi', 6: 'Vascular lesions'
}

print("Loading dataset encoders to ensure perfect metadata formatting...")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Cannot find CSV at {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
df['age'] = df['age'].fillna(df['age'].mean())

# Recreate the exact same encoders used during training
gender_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(df[['sex']])
loc_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(df[['localization']])
age_scaler = StandardScaler().fit(df[['age']])

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_input(img_path, age=50, sex='unknown', loc='unknown'):
    # --- A. IMAGE PREPROCESSING ---
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # NORMALIZE
    img_array = np.expand_dims(img_array, axis=0)

    # --- B. METADATA PREPROCESSING ---
    # Convert inputs to dataframes to match encoder expectations
    age_df = pd.DataFrame({'age': [age]})
    sex_df = pd.DataFrame({'sex': [sex]})
    loc_df = pd.DataFrame({'localization': [loc]})

    # Transform using the exact rules from training
    age_scaled = age_scaler.transform(age_df)
    sex_encoded = gender_enc.transform(sex_df)
    loc_encoded = loc_enc.transform(loc_df)

    # Combine into the final 19-length array
    meta_array = np.hstack([age_scaled, sex_encoded, loc_encoded]).astype('float32')
    
    return img_array, meta_array

def predict_image(image_path, age=50, sex='unknown', loc='unknown'):
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        return

    img_in, meta_in = preprocess_input(image_path, age, sex, loc)
    predictions = model.predict([img_in, meta_in], verbose=0)
    
    print(f"\n========================================")
    print(f"🖼️ Analyzing: {os.path.basename(image_path)}")
    print(f"Patient Data: Age {age}, Sex: {sex}, Loc: {loc}")
    print("========================================")
    
    for i, prob in enumerate(predictions[0]):
        class_name = CLASSES[i]
        print(f"{class_name.ljust(30)}: {prob*100:05.2f}%")
        
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    print(f"\n🏆 PREDICTION: {CLASSES[predicted_class]} ({confidence*100:.2f}%)\n")

if __name__ == "__main__":
    # This checks if you passed an image via the terminal!
    if len(sys.argv) > 1:
        target_image = sys.argv[1]
        # You can pass real metadata here later. Defaulting to 'unknown'
        predict_image(target_image, age=45, sex='male', loc='back')
    else:
        print("⚠️ Please provide an image path. Example: python backend/test_model.py uploads/image3.jpg")