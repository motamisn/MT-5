# **mT5-Base Hindi ↔ Bhili Translation Model**

This repository contains code and resources for fine-tuning the **mT5-base** model for **Hindi-to-Bhili** and **Bhili-to-Hindi** translation. The model is trained using a custom dataset of parallel Hindi-Bhili sentences.

---

## 📖 **Model Details**

- **Base Model**: [google/mt5-base](https://huggingface.co/google/mt5-base)
- **Type**: Sequence-to-sequence (Seq2Seq) transformer
- **Parameters**: ~580M
- **Fine-tuning**: Optimized for translation from **Hindi → Bhili** and **Bhili → Hindi**

---

## 📂 **Dataset**

- **Size**: 133,500 parallel Hindi-Bhili sentence pairs
- **Source**: Custom dataset
- **Usage**: Supports both **Hindi → Bhili** and **Bhili → Hindi** translations

---

## 🚀 **Training Process**

### **1️⃣ Preprocessing**

- **Tokenizer**: `AutoTokenizer.from_pretrained("google/mt5-base")`
- **Max Sequence Length**: 128 tokens
- **Truncation & Padding**: Applied

### **2️⃣ Training Configuration**

| **Hyperparameter**     | **Value**    |
|----------------------|------------|
| Learning Rate       | `1e-4`     |
| Epochs             | `100`      |
| Weight Decay       | `0.01`     |
| Logging Steps      | `100`      |
| Mixed Precision (FP16) | `True`  |
| Warmup Steps       | `500`      |
| Gradient Checkpointing | `True`  |
| Optimizer          | `adamw_torch` |

---

## 📊 **Evaluation Metrics**

The model is evaluated using the following metrics:

| **Metric**  | **Score**  |
|------------|-----------|
| **BLEU**   | `11.21`   |
| **chrF**   | `42.80`   |
| **chrF++** | `40.92`   |

---

### **3️⃣ Loss and Scores Graph**

![Alt text](img/train_loss.png)   ![Alt text](img/validation_loss.png)

** Bleu and Chrf Scores Graph**

![Alt text](img/Bleu.png)   ![Alt text](img/chrf.png)
---

## 📥 **Model Saving & Deployment**

After fine-tuning, the trained model and tokenizer are saved in:

📂 `./mt5-base-hi-bhili-finetuned/`

---

## 🖥️ **Inference Example**

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
 
tokenizer = AutoTokenizer.from_pretrained("./mt5-base-hi-bhili-finetuned")
model = AutoModelForSeq2SeqLM.from_pretrained("./mt5-base-hi-bhili-finetuned")

def translate(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    output = model.generate(**inputs, max_length=128)
    return tokenizer.decode(output[0], skip_special_tokens=True)

print(translate("आपका स्वागत है")) 
```

---

## 🙌 **Acknowledgments**

- **Google** for the `mT5-base` model
- **Hugging Face** for the `Transformers` and `Datasets` library

---


