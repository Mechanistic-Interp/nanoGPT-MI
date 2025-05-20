# nanoGPT-MI

A modified implementation of nanoGPT for mechanistic interpretability tasks.

## Installation

1. Install UV package manager:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create a virtual environment:
```bash
uv venv --python 3.12     
source .venv/bin/activate  # On Linux/macOS
# OR
.venv\Scripts\activate  # On Windows
```

3. Install requirements:
```bash
uv pip install -r requirements.txt
```

## Usage

1. Prepare the GPT-2 weights:
```bash
python prepare_gpt2.py
```
This will save the GPT-2 weights as a `.pt` file in the `out` directory.

2. Capture neuron activations for emotions:
```bash
python neuron_activations/main.py capture --model_path out/gpt2.pt --emotions joy sadness anger --samples_per_emotion 100
```

The captured activations will be stored in `neuron_activations/output/captures/detailed_activations/`.

## Additional Commands

List available neural components:
```bash
python neuron_activations/main.py list --model_path out/gpt2.pt
```

Analyze captured component sizes:
```bash
python neuron_activations/main.py analyze --output_dir neuron_activations/output/captures
```

Visualize activations across emotions:
```bash
python neuron_activations/main.py visualize --input_dir neuron_activations/output/captures
```




