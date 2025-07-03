# Brain-Tumor-Classification-CNN-Streamlit
Brain tumor detection system built using Convolutional Neural Networks (CNNs) and trained on MRI image datasets. The project includes a web-based Streamlit application that predicts tumor types — glioma, meningioma, pituitary, and no tumor — from uploaded brain scan images.


# 📌 Features
- Trained on publicly available brain MRI datasets
- Uses TensorFlow & Keras for building and training the CNN
- Streamlit-powered frontend to upload and classify new images
- Supports image uploads in `.jpg`, `.jpeg`, and `.png` format
- Simple and lightweight architecture ideal for beginners

🚀 Getting Started
1. Create a Virtual Environment
    python -m venv venv
    Activate the environment:
      On Windows:
        venv\Scripts\activate
      On Linux/Mac:
        source venv/bin/activate
2. Install Dependencies
    pip install -r requirements.txt
3. Prepare the Dataset
   Make sure your training data is structured like this: 
    Training/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
    (Optional) You can place test or demo images for manual prediction in:
    dataset/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
4. Train the Model (Optional)
    If you want to train the model yourself:
      python train_model.py
    This will generate and save a model to:
      model/brain_tumor_model.h5
5. Run the Streamlit App
    streamlit run app.py
    Then open http://localhost:8501 in your browser.
Upload a brain MRI image (.jpg, .jpeg, or .png) to get a prediction.


📚 Dataset Used
  Kaggle Brain Tumor Classification Dataset
    https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
