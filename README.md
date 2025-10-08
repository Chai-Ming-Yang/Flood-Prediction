# 🌊 AI Flood Prediction Project

This project uses both Deep Learning (LSTM) and Machine Learning (Random Forest) models to predict flood occurrence based on various datasets. A user-friendly GUI is provided in `main.ipynb` to facilitate easy interaction and prediction.

> **Note:**  
> This project was originally developed outside GitHub and later uploaded to this repository.  
> The following contributors were part of the original development team and are credited for their valuable contributions.

## 👥 Contributors
- **@ambervs (amber)** — Collaborator  
- **@jordanlcr (Lee Chong Ren / Jordan)** — Collaborator  
- **@LeafStardust** — Collaborator  
- **@Vecrex** — Collaborator  
- **Your Name** — Lead Developer & Project Manager  

---

## 🚀 Getting Started

Follow these steps to run the project in **Visual Studio Code (VS Code)**:

### 1. Open the Repository in VS Code

```bash
File > Open Folder
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

**Windows:**
```bash
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Launch the Project

Open `main.ipynb` in VS Code (with the Jupyter extension), and run all cells to start the GUI.

---

## 📁 Repository Structure

```
datasets/
├── A_Flood_Dataset.csv
├── B1_Monthly_Rainfall.csv
└── B2_Monthly_Rainfall.csv
```
Contains all datasets used in this project for training and prediction.

```
src/
├── DL/
│   ├── trained_model/
│   │   ├── DL_verA_trained.keras
│   │   ├── DL_verB1_trained.keras
│   │   └── DL_verB2_trained.keras
│   ├── DL_verA.ipynb
│   ├── DL_verB1.ipynb
│   └── DL_verB2.ipynb
│
├── ML/
│   ├── trained_model/
│   │   ├── ML_verA_trained.pkl
│   │   ├── ML_verB1_trained.pkl
│   │   └── ML_verB2_trained.pkl
│   ├── ML_verA.ipynb
│   ├── ML_verB1.ipynb
│   └── ML_verB2.ipynb
```

- `DL/`: Contains LSTM-based models. Each notebook performs preprocessing, training, saving, and evaluation.
- `ML/`: Contains RandomForest-based models. Each notebook also includes preprocessing, training, saving, and evaluation steps.
- The `trained_model/` folders hold the serialized trained models for each approach and dataset version.

```
main.ipynb
```
A ready-to-use, interactive notebook with GUI to make predictions using the pre-trained ML models.

---

## 🧠 How `main.ipynb` Works

### Dataset Selection GUI
- Lets the user pick a dataset (A, B1, or B2).
- Displays the first few rows (`head()`) of the selected dataset.
- Includes a button to confirm which ML model (trained on the selected dataset) will be used for prediction.

### Prediction Form GUI
- Based on the confirmed ML model, displays a form with all necessary input features.
- User fills in the form, and the model processes the inputs.
- The model then predicts the flood status for the next time period (month/year, depending on the dataset).

---

## 📦 Model Saving & Loading

Below are examples of how to save and load trained models used in this project:

### ML (Random Forest using `joblib`)

```python
import joblib

# Save model
joblib.dump(myModel, 'path/to/model.pkl')

# Load model
myModel = joblib.load('path/to/model.pkl')
```

### DL (LSTM using Keras)

```python
from tensorflow.keras.models import load_model

# Save model
model.save('path/to/model.keras')

# Load model
model = load_model('path/to/model.keras')
```

---

## 📄 Other Files

- `requirements.txt`: All Python dependencies required to run the project.
- `REFERENCE.md`: Contains reference material or sources used during model development.

---

## 🛠 Technologies Used

- Python (Jupyter Notebooks)
- Pandas, Scikit-learn, TensorFlow/Keras
- ipywidgets (for interactive GUIs)
- VS Code (recommended IDE)

---

## 📬 Feedback

Feel free to open issues or pull requests if you find bugs or want to contribute!



