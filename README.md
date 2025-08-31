# Plant Disease Classifier

A deep learning project that classifies plant diseases from leaf images using Convolutional Neural Networks (CNNs). This tool helps farmers and researchers quickly identify plant diseases for early intervention.

## How It Works
1. **Input Images:** Provide images of plant leaves.  
2. **Preprocessing:** Images are resized and normalized to fit the model input.  
3. **CNN Model:** A Convolutional Neural Network extracts features from leaf images, like texture, color, and spots.  
4. **Prediction:** The model predicts the disease class for each image.  
5. **Output:** The predicted disease label helps identify the issue so proper action can be taken.

The project supports:
- Training a CNN from scratch (`train_model.py`)  
- Using transfer learning with a pretrained model (`train_transfer.py`)  
- Making predictions on new images (`predict.py`)  

## Dataset
The model uses the **Plant Disease Dataset** from Kaggle: [https://www.kaggle.com/datasets/emmarex/plantdisease](https://www.kaggle.com/datasets/emmarex/plantdisease)

- Contains thousands of labeled images of plant leaves.  
- Includes multiple crops and diseases.  
- Images are organized in folders by disease class.  

**Note:** The dataset is not included in this repository due to size restrictions. Download it manually from Kaggle.
