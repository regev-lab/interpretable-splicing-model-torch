PyTorch implementation of pre-trained splicing model from "Deciphering RNA splicing logic with interpretable machine learning" (Liao et al., 2023). The manuscript is available [here](https://www.pnas.org/doi/abs/10.1073/pnas.2221165120). Train, test sequences are provided in the `data` directory.

## Requirements

Python-side requirements:

- Python 3.9+
- `pandas`
- `numpy`
- `torch`

Install them with:

```bash
pip install pandas numpy torch
```

System requirement:

- ViennaRNA `RNAfold`

## RNAfold Installation

See the [ViennaRNA GitHub](https://github.com/ViennaRNA/ViennaRNA) for instructions for installing RNAFold. The preprocessing code expects `RNAfold` to be available on `PATH`, unless you pass an explicit `rnafold_bin` path.

## Expected Input Shapes

The model expects channel-first inputs throughout:

- sequence one-hot: `(N, 4, L)`
- wobble: `(N, 1, L)`
- structure: `(N, 3, L)`

In this repository, preprocessing utilities already return arrays in those shapes.

## From CSV to Full Inputs

The canonical input sequence is an unflanked exon in a column named `exon`. Other columns are allowed and are preserved as metadata in the output dataset. If your sequence column has a different name, pass it explicitly.

By default, preprocessing adds the fixed model flanks:

- left flank: `CATCCAGGTT`
- right flank: `CAGGTCTGAC`

That means a 70 nt exon becomes a 90 nt model input.

If you do not want flanks added, use `add_flanks=False` in Python or `--no-flanks` in the CLI. In that case, `L` is just the input sequence length.

### Python API

Use `utils.dataframe_to_dataset()` to convert a pandas dataframe into a dataset dictionary of NumPy arrays:

```python
import pandas as pd

from utils import dataframe_to_dataset

df = pd.DataFrame(
    {
        "exon": ["A" * 70, "C" * 70],
        "sample_id": ["s1", "s2"],
    }
)

dataset = dataframe_to_dataset(df)

print(dataset["seq_oh"].shape)      # (2, 4, 90)
print(dataset["struct_oh"].shape)   # (2, 3, 90)
print(dataset["wobbles"].shape)     # (2, 1, 90)
print(dataset["structure"].shape)   # (2,)
print(dataset["mfe"].shape)         # (2,)
```

`dataset["seq_oh"]`, `dataset["struct_oh"]`, and `dataset["wobbles"]` are NumPy arrays. Convert them to torch tensors before passing them into `PNASModel`. Extra dataframe columns are stored as metadata keys like `meta__sample_id`.

If your sequence column is not named `exon`:

```python
dataset = dataframe_to_dataset(df, sequence_column="sequence")
```

If `RNAfold` is not on `PATH`:

```python
dataset = dataframe_to_dataset(df, rnafold_bin="/path/to/RNAfold")
```

### CLI

You can also prepare a dataset directly from a CSV:

```bash
python prepare_dataset.py \
  --input-csv input.csv \
  --output-path dataset.npz
```

If your sequence column is not `exon`:

```bash
python prepare_dataset.py \
  --input-csv input.csv \
  --output-path dataset.npz \
  --sequence-column sequence
```

Optional RNAfold-related arguments: `--rnafold-bin`, `--temperature`, `--max-bp-span`, and `--commands-file`.

The output is a compressed `.npz` archive containing the same dataset dictionary fields returned by the Python API.

## Running the Model

After preprocessing, convert the NumPy arrays to torch tensors and pass them into `PNASModel.forward()`:

Initialize `PNASModel` with an `input_length` that matches the prepared sequence length `L`. If flanks are added, this length includes the flanking nucleotides.

```python
import torch

from model import PNASModel
from utils import dataframe_to_dataset

dataset = dataframe_to_dataset(...)

x_seq = torch.tensor(dataset["seq_oh"], dtype=torch.float32)
x_struct = torch.tensor(dataset["struct_oh"], dtype=torch.float32)
x_wobble = torch.tensor(dataset["wobbles"], dtype=torch.float32)

model = PNASModel(input_length=x_seq.shape[-1])
state_dict = torch.load("model_weights.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

with torch.no_grad():
    prediction = model(x_seq, x_struct, x_wobble)
```

## Sequence-Only Analysis

To inspect sequence properties such as **SR Balance** and **latent sequence activation**, you can use the model methods `compute_sr_balance`, `compute_sequence_activations()`.

```python
# Get one-hot sequences
x_seq = torch.tensor(dataset["seq_oh"], dtype=torch.float32)

# Compute inclusion, skipping sequence activations
a_incl, a_skip = model.compute_sequence_activations(x_seq, agg="mean")

# Compute SR balance
sr_balance = model.compute_sr_balance(x_seq, agg="mean")
```

## Notes

- The default public preprocessing path assumes unflanked exon input and adds model flanks automatically.
- `load_state_dict()` in `PNASModel` resamples position-bias tensors when checkpoint and runtime input lengths differ.
- `load_weights_from_dict()` is available for loading weights converted from an external TensorFlow/Keras export format.

## Citation

Please cite: Liao, Susan E., Mukund Sudarshan, and Oded Regev. "Deciphering RNA splicing logic with interpretable machine learning." Proceedings of the National Academy of Sciences 120.41 (2023): e2221165120.
