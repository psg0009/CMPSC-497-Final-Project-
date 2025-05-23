
#  Fine-tuning T5-small and BART-base for Environmental Research Generation


This project showcases fine-tuning of `t5-small` and `facebook/bart-base` models to automatically generate research methodologies from concise climate science problem statements.  
It combines natural language generation (NLG) with environmental reasoning.

---

##  Project Overview
- **Dataset:** Adapted ClimateFever dataset (~500 curated examples)
- **Models Used:**  
  - `t5-small` → lightweight and fast prototyping  
  - `facebook/bart-base` → rich, fluent generation
- **Tech Stack:** Hugging Face `transformers`, `Trainer` API, Google Colab (Tesla T4 GPU)

---

##  Repository Contents
- `t5.ipynb` — Fine-tuning T5-small
- `Facebookmodel.ipynb` — Fine-tuning BART-base
- `requirements.txt`
- `environmental_science_climatefever_dataset.json` - Dataset samples (JSON)
- Hyperparameter tuning scripts (Optuna)
- `evaluation_results.csv` - Evaluation metrics and results (ROUGE-L, BERTScore)
  
---

## 📈 Key Results
| Model             | ROUGE-L | BERTScore | Notes                                |
|-------------------|---------|-----------|-------------------------------------|
| T5-small           | 0.421   | 0.824     | Best for lightweight prototyping   |
| facebook/bart-base | 0.2068  | 0.8510    | Best for semantic richness         |

 **Recommendation:**  
- Use **BART-base** for higher-quality, semantic outputs.  
- Use **T5-small** when training time or resources are limited.

---

##  Quick Start
```bash
# Install dependencies
pip install requirements.txt

# Clone the repository
git clone https://github.com/psg0009/CMPSC-497-Final-Project-.git
cd https://github.com/psg0009/CMPSC-497-Final-Project-

# Run the notebooks on Google Colab or locally
```

---

##  Team
- Parth Gosar  
- Sahil Pardasani  


---


