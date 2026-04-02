import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.preprocessing import label_binarize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "skin_disease_prediction.h5")
CSV_PATH = os.path.join(BASE_DIR, "dataset", "HAM10000_metadata.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "dataset", "all_images")

print("Loading AI Model for Testing...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

CLASSES = {
    0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df',
    4: 'mel', 5: 'nv', 6: 'vasc'
}
REVERSE_CLASSES = {v: k for k, v in CLASSES.items()}

GENDER_MAP = {'female': [1, 0, 0], 'male': [0, 1, 0], 'unknown': [0, 0, 1]}
LOC_MAP = {'back': 0, 'face': 1, 'chest': 2, 'upper extremity': 3, 
           'lower extremity': 4, 'trunk': 5, 'abdomen': 6, 'scalp': 7}

def preprocess_metadata(age, gender, localization):
    age_scaled = (float(age) - 51.8) / 16.9
    gender_vec = GENDER_MAP.get(str(gender).lower(), [0, 0, 1])
    loc_vec = np.zeros(15) 
    idx = LOC_MAP.get(str(localization).lower(), 0)
    loc_vec[idx] = 1
    return np.hstack([[age_scaled], gender_vec, loc_vec]).astype('float32')

print("Loading Dataset...")
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=['age'])

# NOTE: Testing 2000 images to prevent RAM crashes. 
# To test all 10,000, delete or comment out the line below.
df = df.sample(n=2000, random_state=42)

test_images = []
test_metadata = []
test_labels = []

print(f"Processing {len(df)} images. This may take a few minutes...")
for index, row in df.iterrows():
    img_id = row['image_id']
    img_path = os.path.join(IMAGE_DIR, f"{img_id}.jpg")
    
    if not os.path.exists(img_path):
        continue
        
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    
    meta = preprocess_metadata(row['age'], row['sex'], row['localization'])
    label = REVERSE_CLASSES.get(row['dx'], 5)
    
    test_images.append(img)
    test_metadata.append(meta)
    test_labels.append(label)

test_images = np.array(test_images)
test_metadata = np.array(test_metadata)
test_labels = np.array(test_labels)

print("Running Predictions...")
y_pred_prob = model.predict([test_images, test_metadata])
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = test_labels

# --- METRICS & ACCURACY ---
total_acc = accuracy_score(y_true, y_pred)
print("\n" + "="*50)
print(f" TOTAL OVERALL ACCURACY: {total_acc * 100:.2f}%")
print("="*50 + "\n")

report = classification_report(y_true, y_pred, target_names=list(CLASSES.values()))
print("Detailed Classification Report:")
print(report)

# --- PLOTTING ---
print("Generating Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=list(CLASSES.values()), 
            yticklabels=list(CLASSES.values()))
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

print("Generating ROC Curve...")
n_classes = len(CLASSES)
y_true_bin = label_binarize(y_true, classes=range(n_classes))

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    if np.sum(y_true_bin[:, i]) > 0:
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    else:
        fpr[i], tpr[i], roc_auc[i] = [0], [0], 0.0

plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta']
for i, color in zip(range(n_classes), colors):
    if roc_auc[i] > 0:
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC {CLASSES[i]} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.close()

print("\nTesting Complete! Check your folder for confusion_matrix.png and roc_curve.png")