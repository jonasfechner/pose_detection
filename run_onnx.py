import cv2
import numpy as np
import yaml
import onnxruntime as ort
import mediapipe as mp
import matplotlib.pyplot as plt

# -------------------- Configuration --------------------

exercise = "push_up"
video_path = "videos/Push-ups_1.mp4"
onnx_model_path = f"weights/onnx/{exercise}_model.onnx"
output_plot_path = f"plots/rep_detection_{exercise}.png"

dim = 26  # 13 keypoints × 2 (x, y)
BIG_KEYPOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
use_normalization = True

# -------------------- Load Parameters --------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

params = config["exercise_configs"][exercise]
waterline = params["waterline"]
min_valley_length = params["min_valley_length"]
max_valley_length = params["max_valley_length"]
min_volume = params["min_volume"]
max_volume = params["max_volume"]

# -------------------- Load ONNX Model --------------------
ort_session = ort.InferenceSession(onnx_model_path)
input_name = ort_session.get_inputs()[0].name
print("✅ ONNX model loaded")

# -------------------- Pose Estimator --------------------
mp_pose = mp.solutions.pose
pose_tracker = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils

# -------------------- Video Processing --------------------
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
aspect_ratio = width / height
display_height = 720
display_width = int(aspect_ratio * display_height)

predictions = []
filtered_valleys = []

rep_count = 0
frame_index = 0
in_valley = False
valley_start = 0
valley_values = []

print("▶️ Starting inference and display...")
while True:
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose_tracker.process(frame_rgb)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark
        keypoints = np.array([[lm.x, lm.y] for lm in landmarks])[BIG_KEYPOINTS]

        if use_normalization:
            x_min, x_max = keypoints[:, 0].min(), keypoints[:, 0].max()
            y_min, y_max = keypoints[:, 1].min(), keypoints[:, 1].max()
            keypoints[:, 0] = (keypoints[:, 0] - x_min) / (x_max - x_min + 1e-6)
            keypoints[:, 1] = (keypoints[:, 1] - y_min) / (y_max - y_min + 1e-6)

        input_np = keypoints.flatten().astype(np.float32).reshape(1, 1, -1)
    else:
        input_np = np.zeros((1, 1, dim), dtype=np.float32)

    # ONNX inference
    ort_inputs = {input_name: input_np}
    ort_outputs = ort_session.run(None, ort_inputs)
    logits = ort_outputs[0]
    confidence = 1 / (1 + np.exp(-logits[0]))  # sigmoid

    predictions.append(confidence)

    # Valley detection
    if confidence < waterline:
        if not in_valley:
            in_valley = True
            valley_start = frame_index
            valley_values = [confidence]
        else:
            valley_values.append(confidence)
    else:
        if in_valley:
            in_valley = False
            valley_end = frame_index
            valley_length = valley_end - valley_start
            valley_volume = np.sum(np.clip(waterline - np.array(valley_values), 0, None))

            if (
                valley_length >= min_valley_length and
                valley_length < max_valley_length and
                min_volume <= valley_volume <= max_volume
            ):
                filtered_valleys.append({
                    "start": valley_start,
                    "end": valley_end,
                    "length": valley_length,
                    "volume": valley_volume
                })
                rep_count += 1
                print(f" Detected valley at frame {valley_start}–{valley_end} | Volume: {valley_volume:.2f} | Length: {valley_length}")

    # Draw annotation
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Draw frame number (top-left)
    cv2.putText(frame, f"Frame: {frame_index}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    # Draw rep count (top-right)
    cv2.putText(frame, f"Reps: {rep_count}", (width - 300, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    frame_resized = cv2.resize(frame, (display_width, display_height))
    cv2.imshow("Live Rep Detection", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_index += 1

cap.release()
cv2.destroyAllWindows()
print(f"✅ Finished. Total frames processed: {len(predictions)}")

# -------------------- Plot Confidence --------------------
plt.figure(figsize=(12, 6))
plt.plot(predictions, label='Confidence', color='black', linewidth=1.5)
plt.axhline(y=waterline, color='red', linestyle='--', label='Waterline')

for valley in filtered_valleys:
    plt.axvspan(valley["start"], valley["end"], color='blue', alpha=0.3)

plt.xlabel("Frame")
plt.ylabel("Confidence")
plt.title(f"Rep Detection Signal: {exercise.replace('_', ' ').title()}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output_plot_path)
print(f"📈 Plot saved to: {output_plot_path}")
