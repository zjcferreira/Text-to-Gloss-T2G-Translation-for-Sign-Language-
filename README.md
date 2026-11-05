# Text-to-Gloss (T2G) Translation for Sign Languages

This repository contains the datasets and code to train, fine-tune, and evaluate deep learning Seq2Seq models for **Text-to-Gloss (T2G)** translation. The project focuses on translating spoken language text into Sign Language glosses, specifically targeting **German to DGS** (Phoenix-14T) and **Portuguese to LIBRAS** (VLibrasBD).

##  Supported Models
The experiments compare three robust architectures optimized for low-resource translation and VRAM constraints:
* **mBART** (`facebook/mbart-large-50-many-to-many-mmt`)
* **NLLB** (`facebook/nllb-200-distilled-600M`)
* **FLAN-T5** (`google/flan-t5-base`)

## Repository Structure

```text
T2G_Codes_Datasets/
│
├── Phoenix-14T-Dataset/        # Original CSV files for Phoenix-14T (German -> DGS)
├── VLibrasBD-Dataset/          # Original CSV files for VLibrasBD (Portuguese -> LIBRAS)
│
├── mBART-Phoenix-14T/          # mBART experiments for Phoenix-14T
│   ├── gen_variants.py         # Data augmentation and preprocessing script
│   ├── phoenix.ipynb           # Main Jupyter Notebook (Training & Inference)
│   └── *.csv                   # Dataset files
│
├── mBART-VLibrasBD/            # mBART experiments for VLibrasBD
│   ├── gen_variants.py
│   ├── vlibrasbd.ipynb
│   └── *.csv
│
└── Other_models/
    ├── FLANT5-Phoenix-14T/     # FLAN-T5 experiments for Phoenix-14T
    ├── FLAN-T5-VLibrasBD/      # FLAN-T5 experiments for VLibrasBD
    ├── NLLB-Phoenix-14T/       # NLLB experiments for Phoenix-14T
    └── NLLB-VLibrasBD/         # NLLB experiments for VLibrasBD
```

##  Environment Setup & Installation

To ensure full reproducibility and avoid out-of-memory (OOM) errors or library conflicts, it is highly recommended to use a Conda environment.

**1. Create and activate the environment:**
```bash
conda create -n t2g_env python=3.10 -y
conda activate t2g_env
```

**2. Install PyTorch (CUDA 12.1 recommended for modern RTX GPUs):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**3. Install requirements:**
```bash
pip install transformers datasets accelerate evaluate sacrebleu pandas numpy jupyterlab tqdm bitsandbytes
```

> **Note on `torchvision` conflict:** If you encounter a `VideoReader` import error from the `datasets` library during training, simply uninstall torchvision (`pip uninstall torchvision -y`) or use the bypass code provided inside the Jupyter Notebooks.

##  Methodology

The notebooks implement a rigorous NLP pipeline tailored for Sign Language generation:
1. **Data Preparation:** The `gen_variants.py` script cleans the datasets, normalizes text (lowercasing), removes duplicates, and generates a strict vocabulary constraint (`gloss_vocab.txt`).
2. **Two-Phase Training:**
   * **Phase 1 (Generalist):** Trains the model using data augmentation and/or robust learning rates with Differential Learning Rates (DLR).
   * **Phase 2 (Expert Fine-Tuning):** A subsequent refinement phase using strictly clean data and a lower learning rate, preventing catastrophic forgetting.
3. **8-bit Optimization:** Models are loaded using `AdamW8bit` (via `bitsandbytes`) and `float16` precision to prevent VRAM explosion while maintaining high batch sizes effectively through gradient accumulation.

##  How to Run

1. Navigate to the desired model folder (e.g., `cd mBART-VLibrasBD`).
2. Run the preprocessing script to generate the `variants/` folder:
   ```bash
   python gen_variants.py
   ```
3. Open the main Jupyter Notebook:
   ```bash
   jupyter lab vlibrasbd.ipynb
   ```
4. Run all cells. The notebook will automatically process the data, train Phase 1, train Phase 2, and finally evaluate the blind test set.

## 📊 Evaluation Metrics
All inferences are evaluated strictly using:
* **BLEUs (1-4):** Computed with `tokenize='none'` to ensure rigorous, space-aware gloss matching (the standard for Sign Language tasks).
* **ROUGE-L:** To measure the longest common sequence and overall fluency.
* Detailed results are saved locally in a `final_metrics_evaluation.csv` file for qualitative analysis.
