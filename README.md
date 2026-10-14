**
# Facial Expression Recognition Using Deep Learning

**CNN | Transfer Learning | Fine-Tuning | Model Optimization | Real-Time Detection**

A deep learning project that recognizes seven facial expression categories from images and live webcam input using PyTorch and ResNet18.

**Developed by:** Noha Al-Shehri & Shahad Abu Shaheen  
**Organization:** Saudi Data & Artificial Intelligence Authority (SDAIA)  
**Dataset:** FER2013

---

## Project Overview

This project developed a Facial Expression Recognition system using the FER2013 dataset to classify facial expressions into seven categories:

**Angry | Disgust | Fear | Happy | Neutral | Sad | Surprise**

The project covers the complete deep learning workflow, including data preprocessing, model training, evaluation, optimization, and deployment.

The goal was to compare different deep learning approaches and identify the most effective model for facial expression classification.

## Technologies Used

- **Programming:** Python
- **Deep Learning:** PyTorch, Torchvision
- **Model Architecture:** CNN, ResNet18
- **Data Processing:** NumPy, Pandas, OpenCV, Pillow
- **Evaluation:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Gradio
- **Development Environment:** Google Colab

## Dataset

The project uses the **FER2013 dataset**, containing grayscale facial images categorized into seven facial expression classes.

Images were resized to 224 × 224 pixels and prepared for deep learning model training and evaluation.

## Models Implemented

Four deep learning approaches were implemented and evaluated:

1. **CNN from Scratch:** A convolutional neural network trained without pretrained weights.
2. **Transfer Learning (Feature Extraction):** Pretrained ResNet18 with a frozen feature extractor and a trained classification layer.
3. **Partial Fine-Tuning:** ResNet18 with selected final layers unfrozen for training.
4. **Full Fine-Tuning:** ResNet18 with all layers trainable to adapt the model to FER2013.

## Model Performance Comparison

All evaluation metrics were calculated on the test dataset, using macro averaging for Precision, Recall, and F1 Score.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| CNN from Scratch | 61.34% | 50.95% | 51.13% | 50.16% |
| Transfer Learning | 45.67% | 44.88% | 38.18% | 37.73% |
| Partial Fine-Tuning | 60.24% | 58.15% | 54.67% | 55.62% |
| **Full Fine-Tuning** | **68.45%** | **68.41%** | **65.78%** | **66.66%** |

### Model Comparison Visualization

<p align="center">
  <img src="images/images/model_comparison.png" alt="Comparison of four deep learning models" width="780">
</p>

## Best Performing Model

**ResNet18 with Full Fine-Tuning** achieved the highest test accuracy of **68.45%**, outperforming the other three approaches.

However, the training and validation results indicated some overfitting, suggesting that further generalization improvements may be beneficial.

### Confusion Matrix

<p align="center">
  <img src="images/images/confusion_matrix.png" alt="Confusion matrix of the optimized FP16 model" width="620">
</p>

The model performed particularly well on Happy and Surprise expressions. Fear was more difficult to recognize, and some confusion occurred between Sad and Neutral.

### CNN Training and Validation Performance

<p align="center">
  <img src="images/images/training_curves_cnn.png" alt="CNN training and validation curves" width="780">
</p>

## Model Optimization

FP16 optimization was applied to the best-performing model without additional training.

| Metric | Original FP32 | Optimized FP16 |
|---|---:|---:|
| Model Size | 42.73 MB | 21.38 MB |
| Inference Time | 2.57 ms | 3.04 ms |
| Accuracy | 68.45% | 68.45% |
| Precision | 68.41% | 68.42% |
| Recall | 65.78% | 65.79% |
| F1 Score | 66.66% | 66.67% |

**Size reduction: 49.95%**

FP16 significantly reduced model storage requirements while maintaining approximately the same prediction accuracy. However, FP32 provided faster inference on the tested GPU.

### Optimization Comparison

<p align="center">
  <img src="images/images/optimization_comparison.png" alt="FP32 versus FP16 inference speed comparison" width="700">
</p>

## Model Deployment

The trained model was successfully deployed for facial expression prediction using:

- Uploaded facial images
- Face detection
- Live webcam streaming through Gradio

The system displays predicted facial expression labels and model confidence scores.

### Image Prediction Results

The model was tested using four facial images, producing the following predictions:

| Predicted Expression | Confidence |
|---|---:|
| Neutral | 96.34% |
| Sad | 98.55% |
| Angry | 92.57% |
| Happy | 99.94% |

<p align="center">
  <img src="images/images/prediction_results.png" alt="Facial expression predictions on test images" width="780">
</p>

### Real-Time Webcam Detection

A Gradio interface was implemented to detect faces and classify facial expressions from live webcam input.

The system uses the trained ResNet18 model for inference without retraining.

## Challenges and Limitations

- Some facial expressions, particularly Fear, were more difficult to classify accurately.
- Confusion occurred between visually similar expressions, including Sad and Neutral.
- Full Fine-Tuning showed signs of overfitting.
- Predictions on real-world webcam images may be affected by lighting, face orientation, image quality, and differences from the training dataset.
- Facial expressions do not necessarily represent a person's actual emotional state.
- High prediction confidence does not guarantee a correct classification.

## Future Improvements

Future work could explore:

- Stronger data augmentation
- Hyperparameter tuning
- Alternative pretrained architectures
- Improved face detection
- Evaluation on additional real-world datasets

## Final Conclusion

This project successfully demonstrated the application of Convolutional Neural Networks, Transfer Learning, Fine-Tuning, Model Optimization, and Deployment for facial expression recognition.

Among the evaluated approaches, Full Fine-Tuning achieved the best overall performance, while FP16 optimization reduced the model size without a meaningful reduction in test accuracy.

The completed system demonstrates an end-to-end deep learning workflow, from data preprocessing and model training to evaluation, optimization, and live webcam inference.

---

## Project Contributors

**Noha Al-Shehri**  
**Shahad Abu Shaheen**

### Acknowledgment

**Saudi Data & Artificial Intelligence Authority (SDAIA)**

We acknowledge SDAIA in connection with this learning project and the development of practical skills in artificial intelligence and deep learning.

**SDAIA:** https://sdaia.gov.sa/

---

*Built using Python, PyTorch, ResNet18, and Gradio.*
**
