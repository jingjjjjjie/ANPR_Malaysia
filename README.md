# 🧠 YOLO Streamlit App  
*A lightweight and data-driven approach to ANPR (Automatic Number Plate Recognition)*  

---

## ⚙️ Environment Setup  

### 1️⃣ Create and Activate the Conda Environment  

```bash
# Create a new conda environment
conda create -n anpr python=3.9 -y

# Activate the environment
conda activate anpr
```

---

### 2️⃣ Install Dependencies  

#### 🔹 Install PyTorch (CUDA 12.6 compatible)
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

#### 🔹 Install Other Dependencies
Make sure you have a `requirements.txt` file, then run:
```bash
pip install -r requirements.txt
```

---

## 📦 Dataset Setup  

You can obtain the **training**, **testing**, and **validation** datasets from [Roboflow](https://app.roboflow.com/yolo-zmazg?group=LKaPTDt9jTcODXIwefFB).

1. Go to the link above.  
2. Select the **YOLOv8** format.  
3. Download the dataset code and files.  
4. Extract the **images** and **labels** into your project’s dataset directory.  

![Download Dataset](download_dataset.png)

---

## 🚀 Running the App  

Once your environment and datasets are ready:

```bash
# Navigate to the app directory
cd path/to/app

# Run the Streamlit app
streamlit run app.py
```

The app should launch automatically in your default web browser.

---

## 🧩 Features  

- Real-time number plate detection using YOLOv8  
- Interactive and user-friendly Streamlit interface  
- Modular and easy-to-extend architecture  

---

## 📁 Project Structure  

```
YOLO-ANPR/
│
├── app.py                # Streamlit application
├── requirements.txt      # Python dependencies
├── datasets/             # Train/Test/Validation datasets
│   ├── images/
│   └── labels/
└── README.md             # Project documentation
```

---

## 💡 Notes  

- Ensure your GPU drivers are up to date for CUDA compatibility.  
- If CUDA is not available, you can install CPU-only PyTorch instead.  

---

## 🏷️ Credits  

Developed with ❤️ using **YOLOv8**, **Streamlit**, and **PyTorch**.

