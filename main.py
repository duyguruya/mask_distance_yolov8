import cv2
import numpy as np
from ultralytics import YOLO

# --- Model Yükleme ---
person_model = YOLO("yolov8n.pt")
person_model.to("cpu")  # CPU kullanımı zorunlu hâle getirildi

face_model = YOLO("yolov8n-face.pt")
face_model.to("cpu")

# --- Görüntü Yükleme ---
img_path = "crowded.jpg"
img = cv2.imread(img_path)

if img is None:
    print(f"'{img_path}' dosyası yüklenemedi. Dosya yolunu kontrol edin.")
    exit()

print("Görüntü başarıyla okundu. İşlemler başlıyor...")

# --- 1. İnsan Tespiti ---
person_results = person_model(img, classes=[0], verbose=False)
person_boxes = person_results[0].boxes
person_centers = []

if person_boxes is not None and len(person_boxes) > 0:
    for box in person_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        person_centers.append((cx, cy))

person_centers = np.array(person_centers, dtype=np.int32)

# --- 2. Sosyal Mesafe Kontrolü ---
if len(person_centers) > 1:
    for i in range(len(person_centers)):
        for j in range(i + 1, len(person_centers)):
            p1, p2 = person_centers[i], person_centers[j]
            distance = np.linalg.norm(p1 - p2)
            is_safe = distance > 100
            color = (0, 255, 0) if is_safe else (0, 0, 255)
            cv2.line(img, tuple(p1), tuple(p2), color, 2)
            cv2.circle(img, tuple(p1), 5, color, -1)
            cv2.circle(img, tuple(p2), 5, color, -1)

# --- 3. Maske Tespiti ---
face_results = face_model(img, verbose=False)
face_boxes = face_results[0].boxes

if face_boxes is not None and len(face_boxes) > 0:
    for box in face_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, "No Mask", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# --- Sonucu Göster ve Kaydet ---
cv2.imshow("Sonuç", img)
cv2.imwrite("sonuc.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
