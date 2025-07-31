# face-gan-utkface
A custom GAN model trained on the UTKFace dataset to generate synthetic human face images using TensorFlow.

# 🎨 Custom GAN for Face Generation using UTKFace Dataset

This project implements a custom **Generative Adversarial Network (GAN)** using TensorFlow/Keras to generate synthetic human face images. The model is trained on the [UTKFace Dataset](https://www.kaggle.com/datasets/jangedoo/utkface-new), which contains over 20,000 facial images labeled by age, gender, and ethnicity.

---

## 📌 Project Highlights

- 🧠 Implemented both **Generator** and **Discriminator** models using Convolutional Neural Networks (CNNs).
- 🛠️ Designed a **custom training loop** using `tf.GradientTape` and adversarial loss with `BinaryCrossentropy`.
- 🔁 Trained the GAN using latent noise vectors to generate increasingly realistic face images.
- 📷 Visualized generated outputs during training to monitor improvements.
- 💾 Saved and reloaded model weights for both Generator and Discriminator.

---

## 🗃️ Dataset

- **Name:** UTKFace
- **Source:** [Kaggle - UTKFace](https://www.kaggle.com/datasets/jangedoo/utkface-new)
- **Content:** 20,000+ cropped and aligned face images
- **Usage:** Only images used; labels not used in GAN training

---

## 🔄 Training Info

This GAN was trained for **100 epochs** on the UTKFace dataset.  
The training was going well, and the generator was starting to produce realistic outputs.

> 🔁 **Note:** To achieve better quality results, it's recommended to train the model for **more epochs (e.g., 300–500)** depending on GPU capacity and visual output quality.

You can continue training using the `train()` function and save/load weights as needed.



## 🚀 How to Run

1. Clone this repository:
    ```bash
    git clone https://github.com/ritikkalyan1000/face-gan-utkface.git
    cd face-gan-utkface
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Open the notebook:
    - Run `UTKFace_GAN.ipynb` in Jupyter or Colab
    - Upload your `kaggle.json` file when prompted
    - Train the model by running all cells

4. Load saved models (optional):
    ```python
    generator.load_weights('generator_weights.h5')
    discriminator.load_weights('discriminator_weights.h5')
    ```

---

## 📦 Project Structure
face-gan-utkface/
│
├── UTKFace_GAN.ipynb # Main Jupyter notebook
├── generator_weights.h5 # Trained Generator weights
├── discriminator_weights.h5 # Trained Discriminator weights
├── requirements.txt # All dependencies


## 🧰 Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- OpenCV

---

## ✨ Author

**Ritik Kalyan**  
[GitHub: ritikkalyan1000](https://github.com/ritikkalyan1000)

---
