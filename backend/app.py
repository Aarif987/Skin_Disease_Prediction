import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import cv2

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "skin_disease_prediction.h5")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

print("Loading Multimodal AI Model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

CLASSES = {
    0: 'Actinic keratoses', 1: 'Basal cell carcinoma', 
    2: 'Benign keratosis-like lesions', 3: 'Dermatofibroma',
    4: 'Melanoma', 5: 'Melanocytic nevi', 6: 'Vascular lesions'
}

GENDER_MAP = {'female': [1, 0, 0], 'male': [0, 1, 0], 'unknown': [0, 0, 1]}
LOC_MAP = {'back': 0, 'face': 1, 'chest': 2, 'upper extremity': 3, 
           'lower extremity': 4, 'trunk': 5, 'abdomen': 6, 'scalp': 7}

def preprocess_metadata(age, gender, localization):
    age_scaled = (float(age) - 51.8) / 16.9
    gender_vec = GENDER_MAP.get(gender.lower(), [0, 0, 1])
    loc_vec = np.zeros(15) 
    idx = LOC_MAP.get(localization.lower(), 0)
    loc_vec[idx] = 1
    return np.hstack([[age_scaled], gender_vec, loc_vec]).astype('float32')

def is_human_skin(filepath):
    try:
        img = cv2.imread(filepath)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 10, 30], dtype=np.uint8)
        upper1 = np.array([30, 255, 255], dtype=np.uint8)
        lower2 = np.array([160, 10, 30], dtype=np.uint8)
        upper2 = np.array([179, 255, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        skin_percentage = (np.sum(mask > 0) / mask.size) * 100
        return skin_percentage >= 5.0
    except:
        return True

def get_gradcam_heatmap(img_array, meta_array, model):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer("Conv_1").output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([img_array, meta_array])
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        if isinstance(conv_outputs, (list, tuple)):
            conv_outputs = conv_outputs[0]
        predictions = tf.convert_to_tensor(predictions)
        conv_outputs = tf.convert_to_tensor(conv_outputs)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    age = request.form.get('age', 50)
    gender = request.form.get('gender', 'male')
    localization = request.form.get('localization', 'back')

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    if not is_human_skin(filepath):
        return jsonify({'diagnosis': 'Invalid Image', 'confidence': '0%', 'error_message': 'Please upload a valid image of a skin lesion.', 'heatmap_url': None})

    img = load_img(filepath, target_size=(224, 224))
    img_array = np.expand_dims(img_to_array(img) / 255.0, axis=0)
    meta_array = np.expand_dims(preprocess_metadata(age, gender, localization), axis=0)

    predictions = model.predict([img_array, meta_array])
    pred_idx = np.argmax(predictions[0])
    diagnosis = CLASSES[pred_idx]
    confidence = float(predictions[0][pred_idx])

    heatmap_url = None

    if confidence < 0.70:
        diagnosis = 'No disease detected'

    if confidence >= 0.70:
        try:
            heatmap = get_gradcam_heatmap(img_array, meta_array, model)
            img_cv = cv2.imread(filepath)
            heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)
            heatmap_filename = "heatmap_" + file.filename
            cv2.imwrite(os.path.join(UPLOAD_FOLDER, heatmap_filename), superimposed_img)
            heatmap_url = f"http://127.0.0.1:5000/uploads/{heatmap_filename}"
        except Exception as e:
            print(f"Grad-CAM Error: {e}")
            heatmap_url = None

    return jsonify({
        'diagnosis': diagnosis,
        'confidence': f"{confidence*100:.2f}%",
        'heatmap_url': heatmap_url
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)