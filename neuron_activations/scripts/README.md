# Neuron Activations Scripts

This directory contains scripts for extracting, analyzing, and modifying neuron activations in GPT-2 models. Below are instructions and examples for running each script. All commands default to using the CPU.

---

## 1. `create_new_model.py`
**Purpose:** Modifies a GPT-2 model's biases using differences between two sets of neuron activation vectors.

**Usage:**
```bash
python create_new_model.py \
  --model_path ../../out/gpt2.pt \
  --output_path <output_model.pt> \
  --target_data <target_dir> \
  --neutral_data <neutral_dir> \
  [--scaling_factor 0.2] \
  [--limit_to <float>] \
  [--component_types proj ln] \
```

**Example:**
```bash
python create_new_model.py \
  --model_path ../../out/gpt2.pt \
  --output_path ./modified_model.pt \
  --target_data ../data/religion \
  --neutral_data ../data/neutral \
  --scaling_factor 0.2 \
  --component_types proj ln \
  --device cpu
```

---

## 2. `get_neuronal_activations.py`
**Purpose:** Computes average neuron activations for a dataset and saves them.

**Usage:**
```bash
python get_neuronal_activations.py \
  --model_path ../../out/gpt2.pt \
  --output_dir <output_dir> \
  --parquet_file <data.parquet> \
  [--output_name <prefix>] \
  [--batch_size 32] \
  [--checkpoint_interval 100]
```

**Example:**
```bash
python get_neuronal_activations.py \
  --model_path ../../out/gpt2.pt \
  --output_dir ../data/religion \
  --parquet_file ../../data/neuronal_parquets/religion.parquet \
  --output_name religion \
  --batch_size 32 \
  --checkpoint_interval 100 \
  --device cpu
```

---

## 3. `run_model.py`
**Purpose:** Interactive CLI for generating text with a (possibly modified) GPT-2 model.

**Usage:**
```bash
python run_model.py \
  --model_path ../../out/gpt2.pt \
  [--max_tokens 100] \
  [--temperature 0.8] \
  [--top_k 40]
```

**Example:**
```bash
python run_model.py \
  --model_path ../../out/gpt2.pt \
  --max_tokens 100 \
  --temperature 0.8 \
  --top_k 40 \
  --device cpu
```

---

## 4. `generate_neutral_dataset.py`
**Purpose:** Extracts first sentences from the OpenWebText dataset and saves them as a neutral dataset in Parquet format.

**Usage:**
```bash
python generate_neutral_dataset.py
```

**Example:**
```bash
python generate_neutral_dataset.py
```
- Output: `openwebtext_first_sentences.parquet` in the current directory.

---

**Notes:**
- All scripts default to CPU. If you want to use GPU, add `--device cuda` where supported.
- Adjust file paths as needed for your environment.
- For custom datasets, adapt `get_neuronal_activations.py` to your data format. 