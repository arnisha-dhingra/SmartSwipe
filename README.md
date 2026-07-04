# SmartSwipe: Active Learning for Photo Decluttering

![SmartSwipe](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-red)

**SmartSwipe** is a human-in-the-loop application designed to declutter large photo libraries using minimal manual labeling. It was developed during a 6-week project under the Association for Computing Activities (ACA), IIT Kanpur (May'26 - July'26).

##  The Concept

Your phone holds 5,000 photos. Reviewing them one-by-one is agonizing — and nobody has the time. 
Instead of manually labeling thousands of images, you build an app where you only swipe on **~300 photos**. A Tinder-style UI connects to an **Active Learning** backend that automatically handles the remaining 4,700 images based on your taste.

##  Objective

Design a human-in-the-loop application to declutter large photo libraries using minimal manual labeling.

##  Approach & Technical Stack

- **Transfer-learning Pipeline**: Engineered with a **MobileNetV2** backbone for **1,280-dim** semantic feature extraction.
- **Active Learning**: Implemented via **uncertainty sampling**; retrains Logistic Regression on high-entropy (P=0.5) samples.
- **Frontend**: Built a full-stack **Streamlit** app with persistent session-state and non-destructive review to optimize labeling throughput.

### Technologies Used
- **Python 3.10+**: Core language for all project logic and data processing.
- **PyTorch & torchvision**: Extracts high-fidelity semantic embeddings via frozen MobileNet V2 weights.
- **Streamlit**: Fast, interactive web frontend — zero complex frontend knowledge needed.
- **NumPy, Pandas & Matplotlib**: Matrix math for the from-scratch classifier, dataset management, and visualization.
- **Local Export**: Saves decisions to disk and exports predicted delete candidates — no cloud APIs required.

##  Impact & Results

- **Auto-classified a 2,000-image library** after labeling only **300 photos (15%)**; slashing manual annotation effort by **85%**.
- **Verified predictive accuracy of 94%** via held-out validation, confirming the uncertainty-sampling strategy convergence.

##  How It Works: The AI Pipeline

1. **Feature Extraction**: The model learns your preferences from just a handful of swipes.
2. **Preference Classifier**: Uses Logistic Regression to determine what you like to keep vs delete.
3. **Uncertainty Sampling**: Intelligently prioritizes the images it's least certain about — maximizing learning efficiency with minimal human effort.

##  Installation and Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/arnisha-dhingra/SmartSwipe.git
   cd SmartSwipe
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 👥 Mentors
Special thanks to the project mentors from ACA, IIT Kanpur:
- Yashjeet Singh
- Kush Bhartiya
- Vansh Bajaj
- Yash Dabi
