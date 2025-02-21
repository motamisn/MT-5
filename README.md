# **mT5 Hindi ↔ Bhili Translation Models**

This repository contains code and resources for fine-tuning the **mT5-base** and **mT5-small** models for **Hindi-to-Bhili** and **Bhili-to-Hindi** translation. These models have been trained on a custom dataset of parallel Hindi-Bhili sentences.

---

## 📖 **Model Details**

### **1️⃣ mT5-Base**
- **Base Model**: [google/mt5-base](https://huggingface.co/google/mt5-base)
- **Type**: Sequence-to-sequence (Seq2Seq) transformer
- **Parameters**: ~580M
- **Fine-tuning**: Optimized for Hindi ↔ Bhili translation

### **2️⃣ mT5-Small**
- **Base Model**: [google/mt5-small](https://huggingface.co/google/mt5-small)
- **Type**: Sequence-to-sequence (Seq2Seq) transformer
- **Parameters**: ~300M
- **Fine-tuning**: Optimized for Hindi ↔ Bhili translation

---

## 📂 **Dataset**

- **Size**: 133,500 parallel Hindi-Bhili sentence pairs
- **Source**: Custom dataset
- **Usage**: Supports both **Hindi → Bhili** and **Bhili → Hindi** translations

---

## 🚀 **Training Process**

### ** Preprocessing**
- **Tokenizer**: `AutoTokenizer.from_pretrained("google/mt5-small")` or `AutoTokenizer.from_pretrained("google/mt5-base")`
- **Max Sequence Length**: 128 tokens
- **Truncation & Padding**: Applied

## 📊 **Evaluation Metrics**

### **mT5-Base Performance**
| **Metric**  | **Score**  |
|------------|-----------|
| **BLEU**   | `11.21`   |
| **chrF**   | `42.80`   |
| **chrF++** | `40.92`   |

### **mT5-Small Performance**
| **Metric**  | **Score**  |
|------------|-----------|
| **BLEU**   | `10.45`   |
| **chrF**   | `41.60`   |
| **chrF++** | `40.05`   |

---

## 📥 **Model Saving & Deployment**

After fine-tuning, the trained models and tokenizers are saved in:

📂 `./mt5-small-hi-bhili-finetuned/`  *(for mT5-Small)*  
📂 `./mt5-base-hi-bhili-finetuned/`  *(for mT5-Base)*  

---

## 🖥️ **Inference Example**

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return tokenizer, model

def translate(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    output = model.generate(**inputs, max_length=128)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Load mT5-Small model
tokenizer, model = load_model("./mt5-small-hi-bhili-finetuned")
print(translate("आपका स्वागत है", tokenizer, model))  # Expected Output: "तुमारो सुगतो चे"

# Load mT5-Base model
tokenizer, model = load_model("./mt5-base-hi-bhili-finetuned")
print(translate("आपका स्वागत है", tokenizer, model))
```

---

## 🙌 **Acknowledgments**

- **Google** for the `mT5` models
- **Hugging Face** for the `Transformers` and `Datasets` library

---
