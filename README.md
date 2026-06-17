# SSAF-Net

SSAF-Net is a PyTorch implementation for hyperspectral unmixing. The current
repository includes a Jasper Ridge dataset example and trains the SSAF model from
`main.py`.

## Project Structure

```text
.
├── dataset/
│   ├── JasperRidge2_R198.mat
│   └── JasperRidge2_end4.mat
├── model/
│   └── SSAF.py
├── utils/
│   ├── FCLSU.py
│   ├── hyperVca.py
│   ├── loadhsi.py
│   └── result_em.py
└── main.py
```

## Requirements

Install the Python packages used by the project:

```bash
pip install numpy scipy torch tqdm
```

If you want to train with GPU acceleration, install a CUDA-enabled PyTorch build
that matches your CUDA version from the official PyTorch installation guide.

## Dataset

The included loader currently supports the `ridge` case and expects these files:

```text
dataset/JasperRidge2_R198.mat
dataset/JasperRidge2_end4.mat
```

The default settings in `main.py` use:

```python
case = "ridge"
rCol = 100
nCol = 100
epochs = 2000
z_dim = 4
```

## Run

From the project root, run:

```bash
python main.py
```

The script automatically uses CUDA when available:

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

After training, it prints elapsed time and evaluation metrics:

```text
aRMSE_Y
aSAD_Y
aRMSE_a
aRMSE_M
aSAD_em
```

## Training Parameters

The `train` function in `main.py` exposes these hyperparameters:

```python
train(
    lr=0.005,
    lambda_y2=0.04,
    lambda_kl=0.001,
    lambda_pre=10,
    lambda_sad=5,
    lambda_vol=10,
)
```

You can edit the defaults directly in `main.py`, or call `train(...)` with
different values from another script.

## Notes

- `main.py` defines `cases = ["ex2", "ridge", "houston"]`, but
  `utils/loadhsi.py` currently implements only the `ridge` branch.
- Training for `epochs = 2000` can take a long time, especially on CPU.
- The project does not currently save model checkpoints or output maps by
  default; it only prints the final metrics.
