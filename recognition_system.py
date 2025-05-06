import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# ----------------------------
# Define the Focal Loss again (same as during training)
def focal_loss(gamma=2.0, alpha=0.25):
    alpha_tensor = tf.constant([1.0, 2.0, 1.5, 0.6, 0.8, 0.8, 1.2])  # Adjusted for 7 emotion classes
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        ce = -y_true * tf.math.log(y_pred)
        weight = tf.math.pow(1 - y_pred, gamma) * alpha_tensor
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=-1))
    return loss_fn
# ----------------------------

# ----------------------------
# Load the model
model = load_model("./emotion_cnn_model_v1.keras", custom_objects={'loss_fn': focal_loss()})
# ----------------------------

# ----------------------------
# Define emotion labels
emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
# ----------------------------

# ----------------------------
# Start Webcam
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # CAP_AVFOUNDATION is better for Mac

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting...")
        break

    # --------------------------------------
    # Preprocessing for model
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    resized = cv2.resize(gray, (48, 48))            # Resize to 48x48 (model input size)
    normalized = resized / 255.0                    # Normalize pixel values to [0, 1]
    input_data = np.expand_dims(normalized, axis=0) # Add batch dimension
    input_data = np.expand_dims(input_data, axis=-1) # Add channel dimension (1 channel)

    # Predict
    preds = model.predict(input_data, verbose=0)
    emotion_idx = np.argmax(preds)
    emotion_label = emotion_classes[emotion_idx]
    # --------------------------------------

    # Draw prediction on the frame
    cv2.putText(frame, f'Emotion: {emotion_label}', (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Emotion Recognition - Press q to Quit', frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
