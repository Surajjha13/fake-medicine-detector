
# Fake Medicine Detector (Hybrid CNN + OCR + API Verification)

A hybrid deep learning system that identifies **Real vs Fake medicine packaging** using:

✔ Convolutional Neural Network (VGG16)
✔ OCR text extraction (Tesseract)
✔ Multi-API verification (OpenFDA, RxNorm, HealthOS)
✔ Hybrid decision engine
✔ Streamlit web application

This project helps detect counterfeit medicines using a combination of **computer vision**, **text recognition**, and **official medical database cross-verification**.

---

## Features

### **1. CNN-Based Visual Classification**

* Uses transfer learning with **VGG16**
* Classifies images into **Real** or **Fake**
* Achieves **96.44% accuracy**, **95.32% precision**, **99.30% recall**

### **2. OCR Text Extraction**

* Reads brand names and text printed on packaging
* Uses **Tesseract OCR**
* Cleans extracted text for API matching

### **3. Multi-API Verification**

Verifies extracted medicine names using:

* **OpenFDA API**
* **RxNorm API**
* **HealthOS API** (India-specific)

Helps confirm if a drug name actually exists in medical records.

### **4. Hybrid Decision Engine**

Combines:

* CNN prediction
* OCR output
* API verification
  to produce a final trustworthy result.

### **5. Streamlit Web App**

User-friendly interface:

* Upload image or use webcam
* View OCR text, API results, CNN prediction
* Get the final Real/Fake decision with confidence bar

---

## Dataset

### Dataset Structure

```
dataset/
├── train/
│   ├── Fake/
│   └── Real/
├── val/
│   ├── Fake/
│   └── Real/
└── test/
    ├── Fake/
    └── Real/
```

### Image Count

| Split      | Fake | Real | Total |
| ---------- | ---- | ---- | ----- |
| Train      | 238  | 423  | 661   |
| Validation | 140  | 313  | 453   |
| Test       | 162  | 287  | 449   |

Dataset sourced from:

* Kaggle public datasets
* Roboflow counterfeit samples
* Additional manually collected images

---

## Model Architecture

* VGG16 (pretrained on ImageNet)
* Global Average Pooling
* Dropout Layer
* Fully connected sigmoid output

**Loss:** Binary Crossentropy
**Optimizer:** Adam
**Metrics:** Accuracy, Precision, Recall

---

## Model Performance

### Test Results

| Metric    | Score      |
| --------- | ---------- |
| Accuracy  | **96.44%** |
| Precision | **95.32%** |
| Recall    | **99.30%** |

### Confusion Matrix

```
[[148, 14],
 [  2, 285]]
```

The model performs extremely well in identifying both classes, especially real medicines.

---

## Tech Stack

* Python
* TensorFlow / Keras
* Tesseract OCR
* Streamlit
* NumPy, Pandas, Scikit-learn
* Requests (API calls)

---

## ▶Running the App

### **1. Clone the repo**

```
git clone https://github.com/your-username/fake-medicine-detector.git
cd fake-medicine-detector
```

### **2. Install dependencies**

```
pip install -r requirements.txt
```

### **3. Run the Streamlit app**

```
streamlit run app.py
```

### **4. Upload a medicine image or use webcam**

The system will output:

* OCR text
* API match info
* CNN class prediction
* Final hybrid decision

---

## Future Enhancements

* Better OCR using Google Vision / EasyOCR
* QR code & batch number verification
* Larger dataset collection
* Mobile app version
* Multi-language OCR
* Fine-tuning deeper VGG16 layers

---

## Acknowledgements

Thanks to:

* Kaggle & Roboflow for datasets
* TensorFlow & Tesseract communities
* OpenFDA, RxNorm & HealthOS for medical APIs

---

## License

This project is for **educational and research purposes only**.

---
