import cv2
import os
import pytesseract
import joblib
import numpy as np

# Set Tesseract path (for Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load trained KNN model
knn = joblib.load('knn_combined_model_csv.pkl')

# Load image
image_path = "p.png"
original_image = cv2.imread(image_path)

# Check if image loaded
if original_image is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")

# Copies for display (separate copies for each method)
image_tesseract = original_image.copy()
image_knn = original_image.copy()

# Preprocess the image for Tesseract and KNN
gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blurred, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 15, 3)

# Image size
h_img, w_img = original_image.shape[:2]

# Tesseract bounding boxes
boxes = pytesseract.image_to_boxes(original_image)

for box in boxes.splitlines():
    b = box.split()
    if len(b) != 6:
        continue  # skip if box doesn't have expected values

    tesseract_char, x1, y1, x2, y2, _ = b
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Flip y-coordinates (OpenCV origin is top-left)
    y1 = h_img - y1
    y2 = h_img - y2

    # --- TESSERACT --- Work on individual image (for Tesseract)
    cv2.rectangle(image_tesseract, (x1, y2), (x2, y1), (0, 255, 0), 2)
    cv2.putText(image_tesseract, tesseract_char, (x1, y2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # --- KNN --- Work on individual image (for KNN)
    if y1 > y2 and x2 > x1:
        char_img = thresh[y2:y1, x1:x2]
        if char_img.size == 0:
            continue
        try:
            resized = cv2.resize(char_img, (28, 28), interpolation=cv2.INTER_AREA)
            flattened = resized.reshape(1, -1).astype(np.float32)
            knn_pred = knn.predict(flattened)[0]

            cv2.rectangle(image_knn, (x1, y2), (x2, y1), (255, 0, 0), 2)
            cv2.putText(image_knn, str(knn_pred), (x1, y2 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        except Exception as e:
            print("Error in KNN prediction:", e)
            continue

# Show the predictions for both methods on separate images
cv2.imshow("Tesseract Prediction", image_tesseract)
cv2.imshow("KNN Prediction", image_knn)

# Save the images if you want to keep them
cv2.imwrite("tesseract_output.jpg", image_tesseract)
cv2.imwrite("knn_output.jpg", image_knn)

cv2.waitKey(0)
cv2.destroyAllWindows()
