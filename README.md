# mask_distance_yolov8

Bu proje, **YOLOv8** tabanlı bir derin öğrenme modeli kullanarak kişilerin **maske takıp takmadığını** ve **sosyal mesafe ihlallerini** gerçek zamanlı olarak tespit etmeyi amaçlar.

---

## Özellikler

- Maske takan / takmayan kişilerin ayrımı
- Sosyal mesafe ihlallerinin hesaplanması
- Gerçek zamanlı video veya canlı kamera akışı üzerinden analiz
- YOLOv8 model ile yüksek doğrulukta nesne algılama

---

## Proje Yapısı

mask_distance_yolov8/
├── crowded.jpg # Örnek test görseli
├── main.py # Ana Python script dosyası
├── README.md # Proje açıklama dosyası
├── yolov8n.pt # Kullanılan YOLOv8 model ağırlığı


---

## Kurulum

1. Bu repoyu klonla:
```bash
git clone https://github.com/duyguruya/mask_distance_yolov8.git
cd mask_distance_yolov8

Gerekli kütüphaneleri yükle (örn. ultralytics, opencv-python):

pip install ultralytics opencv-python

Kullanım
Test etmek için örnek görsel (crowded.jpg) üzerinde çalıştırabilirsin:

python main.py
Not: main.py dosyası içinde doğrudan crowded.jpg ile test yapılmaktadır. Eğer farklı bir görsel ya da video ile test edeceksen kod içinde ilgili kısmı güncellemen gerekir.

Kullanılan Teknolojiler:
Python
YOLOv8 - Ultralytics
OpenCV
Numpy


Her katkı memnuniyetle karşılanır.

İletişim & Bağlantılar:
Geliştirici: Duygu Rüya Çiğ
E-posta: duyguruyacig1403@gmail.com
GitHub: github.com/duyguruya
Proje Linki: github.com/duyguruya/mask_distance_yolov8
LinkedIn: linkedin.com/in/duygu-rüya-çığ-5b7a09322

# mask_distance_yolov8
