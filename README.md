# Tashkeel Transformer 🕌

An Encoder-Decoder Transformer architecture built from scratch in PyTorch for Automatic Arabic Text Diacritization (Tashkeel / الإعراب والتشكيل).

## Architecture Details

- **Type:** Encoder-Decoder Sequence-to-Sequence Transformer
- **Parameters:** ~25.09M
- **Embedding Dimension (`n_embd`):** 384
- **Attention Heads (`n_head`):** 6
- **Layers (`n_layer`):** 6
- **Context Size (`block_size`):** 256
- **Final Validation Loss:** ~0.095

## Key Features

1. **Custom Tashkeel Tokenization:** Handles merged diacritics (e.g., Shaddah combined with Fatha/Kasra/Damma) as single composite vocabulary tokens for higher accuracy.
2. **Bidirectional Encoder Context:** Captures complete sentence semantics before generating diacritics autoregressively.
3. **Pure PyTorch Implementation:** Built without high-level wrappers to ensure full transparency and efficiency.

## Repository Structure

```text
.
├── dataset.py                  # Script to extract input/output sequences from raw parquet data
├── tashkeel_model.py           # Model definition, data preparation, and training loop
├── tashkeel_transformer1.pth   # Saved model weights checkpoint
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation