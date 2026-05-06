# Neuromarketing-Based Advertisement Interest Prediction

A deep learning system that predicts user interest in advertisements using both structured demographic/emotional data and multimodal biosignal + video features.

## 📋 Project Overview

This project develops a robust, efficient, and scalable deep learning system for advertisement interest prediction by combining two complementary paradigms:

1. **Tabular Paradigm:** Leverages demographic and emotional response data from questionnaires
2. **Multimodal Paradigm:** Combines physiological biosignals (BVP, EDA, TEMP, ACC) with facial video data

The system addresses the advertisement interest prediction problem through five complementary model families: **XGBoost**, **LightGBM**, **TabTransformer**, **1D CNN**, **MobileNet**, and a **Fusion model** that combines biosignals and video.

## 🎯 Key Results

| Model | Branch | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|--------|----------|-----------|--------|----------|-----|
| **TabTransformer** | **Tabular** | **82.09%** | **0.84** | **0.82** | **0.83** | **0.8858** |
| LightGBM | Tabular | 80.60% | 0.82 | 0.81 | 0.81 | 0.8594 |
| XGBoost | Tabular | 78.36% | 0.79 | 0.78 | 0.79 | 0.8520 |
| Video (MobileNet) | Multimodal | 68.0% | 0.71 | 0.68 | 0.69 | - |
| Biosignal (1D CNN) | Multimodal | 59.1% | 0.59 | 0.58 | 0.59 | - |
| Fusion | Multimodal | 57.5% | 0.57 | 0.56 | 0.56 | - |

**Key Finding:** The TabTransformer questionnaire-level approach substantially outperforms multimodal approaches, achieving ~14 percentage points higher accuracy than the video-only model and ~25 points higher than fusion.

## 🏗️ System Architecture

### Tabular Branch
```
Raw Demographic & Emotion Data
    ↓
Feature Engineering (34 features)
    ├─→ Age bucketing, emotion aggregation
    ├─→ Positive/negative emotion splits
    └─→ Interaction features (Joy×Surprise, P/N×Joy, etc.)
    ↓
├─ XGBoost (500 estimators, max_depth=6)
├─ LightGBM (histogram-based leaf-wise boosting)
└─ TabTransformer (3 encoder layers, 4 attention heads)
    ↓
Predictions (Interested / Not Interested)
```

### Multimodal Branch
```
Physiological Biosignals (BVP, EDA, TEMP, ACC)          Facial Video
    ↓                                                          ↓
Preprocessing & Resampling (500 timesteps)          Frame Extraction & Resizing
    ↓                                                          ↓
1D CNN (Conv1D→MaxPool→Dense)                    MobileNet (ImageNet pretrained)
    ↓                                                          ↓
64-d Feature Vector ←──────────────────────────────→ 64-d Feature Vector
    ↓                                    ↓
        Feature-Level Fusion (128-d concatenation)
            ↓
        Dense Classifier (128 → 1 neuron, Sigmoid)
            ↓
        Predictions (Interested / Not Interested)
```

## 📊 Dataset

**NeuroBioSense Dataset**
- **Total Samples:** 670 advertisement viewing sessions
- **Class Distribution:** 215 Not Interested, 455 Interested (class imbalance: 1:2.1)
- **Train/Test Split:** 80%/20% stratified (536/134 samples)

**Data Modalities:**
- Participant questionnaire: Demographics (age, gender), emotional annotations (Joy, Surprise, Fear, Disgust, Sadness, Anger), P/N sentiment ratings
- Physiological signals: 32 Hz sampling rate (BVP, EDA, TEMP, ACC X/Y/Z)
- Facial video: Advertisement viewing recordings

## 🛠️ Methodology

### Feature Engineering (Tabular)
- **Demographic Features:** Age bucketing (4 groups), gender, log-transformed time, ad codes
- **Raw Emotions:** 6 emotion percentages extracted from annotations
- **Aggregate Statistics:** Max, mean, sum, count, std, entropy, dominant emotion
- **Derived Features:** Positive/negative ratios, emotion interactions (joy_x_surprise, pn_x_joy), strong_positive indicator

**Total:** 34-dimensional feature vector

### Data Preprocessing
- Forward-fill for merged-cell Excel formatting
- Biosignal segmentation by timestamp with duration-based windowing
- Duration filtering (30–300 seconds) and resampling to 500 timesteps
- Video frame extraction (10 frames/sample) with 224×224 resizing
- Data augmentation (40% probability: horizontal flip, brightness/contrast jitter)
- Class imbalance handling via weighted loss and scale_pos_weight

## 🔧 Installation

### Requirements
```bash
Python 3.8+
PyTorch >= 1.9
TensorFlow >= 2.6
XGBoost >= 1.5
LightGBM >= 3.2
scikit-learn >= 0.24
pandas >= 1.1
NumPy >= 1.19
OpenCV >= 4.5
matplotlib, seaborn
```

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/neuromarketing-ad-interest.git
cd neuromarketing-ad-interest

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
neuromarketing-ad-interest/
├── README.md
├── requirements.txt
├── data/
│   ├── questionnaire.xlsx         # Participant demographics & emotions
│   ├── biosignals.csv             # Physiological data (32 Hz)
│   └── videos/                    # Advertisement video clips
├── src/
│   ├── preprocessing/
│   │   ├── tabular_preprocessing.py
│   │   ├── biosignal_preprocessing.py
│   │   └── video_preprocessing.py
│   ├── models/
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── tabtransformer.py
│   │   ├── biosignal_cnn.py
│   │   ├── video_mobilenet.py
│   │   └── fusion_model.py
│   ├── training/
│   │   ├── train_tabular.py
│   │   ├── train_multimodal.py
│   │   └── utils.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── config.py
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_tabular_pipeline.ipynb
│   └── 03_multimodal_pipeline.ipynb
├── results/
│   ├── models/
│   │   ├── xgboost_model.pkl
│   │   ├── lightgbm_model.pkl
│   │   ├── tabtransformer_checkpoint.pt
│   │   ├── biosignal_cnn.h5
│   │   ├── mobilenet_classifier.h5
│   │   └── fusion_model.h5
│   ├── figures/
│   │   ├── tabtransformer_training_curves.png
│   │   ├── feature_importances_xgboost.png
│   │   └── confusion_matrices.png
│   └── reports/
│       └── evaluation_results.json
└── docs/
    └── project_report.pdf
```

## 🚀 Usage

### 1. Data Preparation
```python
from src.preprocessing import prepare_data

# Preprocess all modalities
X_tabular, X_biosignal, X_video, y = prepare_data(
    questionnaire_path='data/questionnaire.xlsx',
    biosignal_path='data/biosignals.csv',
    video_dir='data/videos/'
)
```

### 2. Train Tabular Models
```python
from src.models import TabTransformer
from src.training import train_tabular

model = TabTransformer(
    input_dim=34,
    d_model=64,
    num_layers=3,
    num_heads=4
)

history = train_tabular(
    model=model,
    X_train=X_train,
    y_train=y_train,
    epochs=150,
    batch_size=32,
    use_cuda=True
)
```

### 3. Train Multimodal Models
```python
from src.models import BiosignalCNN, MobileNetClassifier, FusionModel
from src.training import train_multimodal

fusion_model = FusionModel(
    signal_input_shape=(500, 6),
    video_feature_dim=64
)

history = train_multimodal(
    model=fusion_model,
    X_signal=X_biosignal,
    X_video=X_video,
    y=y,
    epochs=30,
    batch_size=8
)
```

### 4. Evaluation
```python
from src.evaluation import evaluate_all_models

results = evaluate_all_models(
    models=[xgboost, lightgbm, tabtransformer, cnn, mobilenet, fusion],
    X_test=X_test,
    y_test=y_test,
    model_names=['XGBoost', 'LightGBM', 'TabTransformer', 'CNN', 'MobileNet', 'Fusion']
)

print(results)  # Accuracy, Precision, Recall, F1, AUC, Confusion Matrix
```

## 📈 Model Details

### TabTransformer (Best Performing)
- **Input Projection:** Each of 34 features → 64-d embedding via Linear(1, 64)
- **Positional Encoding:** Learnable per-feature encoding to preserve feature identity
- **Encoder:** 3 Transformer layers, 4 attention heads, dim_feedforward=256, dropout=0.3
- **Output:** Global average pooling → MLP head (LayerNorm → Dense(128) → Dense(2))
- **Training:** AdamW (lr=1e-3), CosineAnnealingLR, 150 epochs, best checkpoint by AUC

### 1D CNN (Biosignals)
```
Input (500, 6) 
  → Conv1D(32, k=3, ReLU) → MaxPool(2)
  → Conv1D(64, k=3, ReLU) → MaxPool(2)
  → Flatten → Dense(128, ReLU) → Dropout(0.5)
  → Dense(1, Sigmoid)
```

### MobileNet (Video)
- ImageNet pretrained, first 50 layers frozen, remaining fine-tuned
- Global average pooling → Dense(128, ReLU) → Dropout(0.5) → Dense(64, ReLU)
- Produces 64-d feature vector per sample

## 📖 Novel Contributions

1. **First end-to-end pipeline** for advertisement interest prediction on NeuroBioSense combining tabular and multimodal paradigms
2. **34-dimensional emotion × demographic feature engineering** with rich interaction features
3. **TabTransformer with per-feature embeddings and AUC-based checkpointing** achieving 82.09% accuracy and 0.8858 AUC
4. **Paper-grounded six-way comparison** of gradient boosting, transformers, CNNs, and transfer learning
5. **Fusion-ready architecture** enabling future integration of all three modalities

## ⚠️ Limitations

- Small dataset size (670 samples) limits generalization
- Emotion annotations are self-reported percentages (not objective measurements)
- Signal-to-sample alignment relies on timestamp heuristics
- Subtle facial expressions during naturalistic ad viewing pose challenges
- Binary classification only (no engagement intensity)
- Limited interpretability for deep learning models (TabTransformer, multimodal)

## 🔮 Future Work

- Tri-modal fusion (tabular + biosignal + video) via gated or attention-based mechanisms
- Larger, demographically diverse datasets for improved generalization
- Face detection and landmark preprocessing for enhanced video features
- Multi-level interest prediction (low/medium/high engagement)
- Real-time inference service deployment
- Integration of EEG signals as additional modality
- Cross-validation (StratifiedKFold) for variance estimation

## 👥 Contributors

- **Isha Gupta** (102303007)
- **Mohammad Aaban** (102303015)
- **Kriti Goyal** (102303032)
- **Riddhi Jain** (102483079)

**Advisor:** Dr. Stuti Chug  
**Institution:** Thapar Institute of Engineering & Technology

## 📚 References

[1] Picard et al. (2001) — Physiological signals for emotional intelligence  
[2] Shu et al. (2018) — Emotion recognition using physiological signals  
[3] Kiranyaz et al. (2021) — 1D CNNs for signal classification  
[4] Katsigiannis & Ramzan (2018) — DREAMER dataset for EEG/ECG emotion recognition  
[5] Howard et al. (2017) — MobileNet architecture  
[6] Li & Deng (2022) — Deep facial expression recognition survey  
[7] Donahue et al. (2015) — LRCNs for temporal aggregation  
[8] Poria et al. (2017) — Multimodal fusion strategies  
[9] Soleymani et al. (2012) — Multimodal affect recognition (MAHNOB-HCI)  
[10] Chen & Guestrin (2016) — XGBoost  
[11] Ke et al. (2017) — LightGBM  
[12] Huang et al. (2020) — TabTransformer  

## 📄 License

This project is completed as part of the Deep Learning course (UCS761) at Thapar Institute. Please refer to the institution's guidelines for usage and citation.

## 📧 Contact

For questions or inquiries, please contact the project contributors or submit an issue on GitHub.

---

**Last Updated:** 2024  
**Status:** Complete — Deep Learning Course Project
