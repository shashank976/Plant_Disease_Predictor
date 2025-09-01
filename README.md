# Plant Disease Classifier

A deep learning model that classifies plant leaf images into 16 disease categories using **transfer learning (MobileNetV2)**. Trained on the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) with >33k images.

---

## Features
- Identifies **16 plant leaf conditions** (healthy + diseases).
- Built with **TensorFlow / Keras**.
- Uses **transfer learning (MobileNetV2)** for faster training and higher accuracy.
- Achieves ~85% validation accuracy.
- Includes **training** and **prediction** scripts.

---

## How It Works
1. **Input Images:** Provide images of plant leaves.  
2. **Preprocessing:** Images are resized and normalized to fit the model input.  
3. **CNN Model:** A Convolutional Neural Network extracts features from leaf images, like texture, color, and spots.  
4. **Prediction:** The model predicts the disease class for each image.  
5. **Output:** The predicted disease label helps identify the issue so proper action can be taken.

The project supports:
- Using transfer learning with a pretrained model (`train_transfer.py`)  
- Making predictions on new images (`predict.py`)  

## Dataset
The model uses the **Plant Disease Dataset** from Kaggle: [https://www.kaggle.com/datasets/emmarex/plantdisease](https://www.kaggle.com/datasets/emmarex/plantdisease)

- Contains thousands of labeled images of plant leaves.  
- Includes multiple crops and diseases.  
- Images are organized in folders by disease class.  

**Note:** The dataset is not included in this repository due to size restrictions. Download it manually from Kaggle.
