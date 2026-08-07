# Tashkeel Transformer (سيبويه الغلابة)

An **Encoder-Decoder Seq2Seq Transformer** architecture built entirely from scratch in PyTorch for **Automatic Arabic Text Diacritization (Tashkeel & I'rab / الإعراب والتشكيل)**.

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

---

## Overview & Example

Arabic text without diacritics can often be ambiguous to read or process in downstream NLP tasks. **Tashkeel Transformer** takes unvocalized Arabic sentences and automatically predicts the full set of diacritics (Harakat & Shaddah).

| Input (Without Tashkeel) | Output (With Full Tashkeel) |
| :--- | :--- |
| `تعلموا العربية؛ فإنها من دينكم` | `تَعَلَّمُوا الْعَرَبِيَّةَ؛ فَإِنَّهَا مِنْ دِينِكُمُ` |

---

## Dataset

This project utilizes the **Sadeed Tashkeela Arabic Diacritization Dataset**, a large, high-quality, and clean Arabic diacritized corpus optimized for training and evaluating neural diacritization models.

* **Dataset Link:** [Sadeed_Tashkeela on HuggingFace](https://huggingface.co/datasets/Misraj/Sadeed_Tashkeela)

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/belalfathy211/Tashkeel-Transformer.git
   cd Tashkeel-Transformer
   ```

    

2. **Install required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```


3. **Data Setup (For Training):**
Download the dataset files from HuggingFace and place the `.parquet` files into the `Sadeed_Tashkeela/data/` directory.

---

## How to Use

### Training the Model

To tokenize the dataset and begin training, run:

```bash
python train.py
```

*The best model weights will automatically be saved to `tashkeel_transformer.pth` based on validation loss.*

---

### Interactive Inference (Prediction)

1. **Model Weights:** Download the pretrained weights [`tashkeel_transformer.pth`](https://drive.google.com/file/d/1bjtEgmWXXeMkLJManbuxjsCXUQ3xHTDY/view?usp=sharing) and place the file in the root project directory.
2. **Run the interactive CLI:**
    ```bash
    python predict.py
    ```
3. Type any Arabic sentence into the terminal prompt to view predicted diacritics in real-time.

---

## Model Performance & Results

* **Training Accuracy:** `98.74%`
* **Validation Accuracy:** `98.76%`

---

## 📁 Repository Structure

```text
Tashkeel-Transformer/
├── Sadeed_Tashkeela/
│   └── data/                 # Raw Parquet dataset files
├── transformer/
│   ├── __init__.py
│   ├── modules.py            # Single-Head, Multi-Head & FeedForward layers
│   ├── encoder.py            # EncoderBlock & Encoder architecture
│   ├── decoder.py            # DecoderBlock & Cross-Attention Decoder
│   └── tashkeel_model.py     # Unified Seq2Seq Model & Autoregressive predict()
├── tashkeel_dataset.py       # Dataset reader, character & diacritic tokenizers
├── train.py                  # Training pipeline 
├── predict.py                # Interactive CLI inference script
├── tashkeel_transformer.pth  # Model weights checkpoint
├── requirements.txt          # Python dependencies
└── README.md                 # Documentation

```

---

## Future Improvements & Roadmap

While the current model achieves high token-level accuracy, further enhancements can make it production-ready:

* **Advanced Tokenization:** Move from character-level mapping to Subword BPE or Morphological Segmentation.
* **Expanded Training Data:** Incorporate Classical Arabic (Tafsir, Hadith) and Modern Standard Arabic (MSA) news corpora.
* **Evaluation Metrics:** Compute formal **Diacritic Error Rate (DER)** and **Word Error Rate (WER)** across standard test sets.