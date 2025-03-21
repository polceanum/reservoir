# reservoir: sparsity and scaling experiments

A lightweight library for exploring the sparsity and scaling behaviors of reservoir computing. This repository provides tools for running reservoir computing experiments on both numerical timeseries data (e.g. the Mackey Glass dataset) and text-based datasets (e.g. Wikipedia). It leverages pretrained language models for text embedding extraction, caches both text embeddings and reservoir weights for efficiency, and includes functionality for saving and loading trained readout models.

## Features

- **Reservoir Computing Experiments**: Explore how reservoir size, sparsity (density), and topology (uniform, geometric, smallworld) affect performance.
- **Multi-Dataset Support**: Run experiments on standard numerical timeseries data _or_ text datasets.
  - **Mackey Glass Dataset**: The default numerical dataset provided is a Mackey Glass time series (e.g., `MackeyGlass_t17.txt`).
- **Pretrained Embeddings for Text**: Automatically download and use Hugging Face pretrained models (e.g., `distilbert-base-uncased`) to extract text embeddings.
- **Caching Mechanisms**:
  - **Text Embeddings Caching**: Once computed, text embeddings are saved to a cache file (e.g. `wikipedia_sample.txt.embeddings.pt`) to avoid reprocessing.
  - **Reservoir Weights Caching**: Reservoir weight matrices are cached based on experiment parameters to speed up repeated runs.
- **Model Saving and Loading**: Save the trained readout model and reload it later to skip retraining.
- **Sample Output Printing**:
  - For **text datasets**: Uses a nearest-neighbor decoder (based on cosine similarity) to retrieve a corresponding text sample.
  - For **timeseries data**: Prints raw numerical outputs.
- **Profiling and Visualization**: Optional PyTorch profiling and plotting of reservoir properties and training progress.

## Requirements

- Python 3.7+
- [PyTorch](https://pytorch.org/)
- NumPy
- scikit-learn
- NetworkX
- tqdm
- [Transformers](https://github.com/huggingface/transformers) (`pip install transformers`)
- [Datasets](https://github.com/huggingface/datasets) (for automatic Wikipedia sample download; `pip install datasets`)
- (Optional) Matplotlib for visualization if you enable `--viz`.

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd reservoir
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

*Note: Ensure you have a compatible PyTorch installation (CPU or CUDA) installed.*

## Usage

The main script is located at `src/sparse_reservoir.py` and accepts numerous command-line arguments to configure your experiment.

### Command-Line Arguments

- **Data & Dataset Options**
  - `--data-file`: Path to the input data file. For numerical experiments, you can use the Mackey Glass dataset (e.g., `./data/MackeyGlass_t17.txt`).
  - `--dataset`: Type of dataset: `timeseries` (default) or `wikipedia`.
  - `--pretrained-model`: Pretrained model name for text embeddings (used with `--dataset wikipedia`), e.g., `distilbert-base-uncased`.

- **Training Options**
  - `--fp`: Floating point precision (choices: 16, 32, 64; default is 64).
  - `--lr`: Learning rate for training.
  - `--opt`: Optimizer (`adam`, `adamw`, `adagrad`, `rprop`, `rmsprop`, or `lr`).
  - `--epochs`: Number of training epochs.

- **Reservoir Options**
  - `--top`: Reservoir topology (`uniform`, `geometric`, or `smallworld`).
  - `--dim-res`: Reservoir size (number of reservoir nodes).
  - `--rho`: Reservoir density.
  - `--alpha`: Reservoir leak rate.
  - `--rest`: Enable reservoir spectral radius estimation.

- **Valve and Data Dimensions**
  - `--valve-in` / `--valve-out`: Input and output valve sizes.
  - `--dim-in` / `--dim-out`: Input and output dimensions.  
    For text datasets, these are automatically set to match the pretrained model’s embedding size.

- **Visualization & Profiling**
  - `--viz`: Enable visualization of reservoir properties.
  - `--profile`: Enable PyTorch profiling for reservoir and readout training.

- **Sample Output & Model Persistence**
  - `--print-samples`: Print a few generated sample outputs.
    - For `wikipedia` mode, this decodes outputs using nearest-neighbor matching.
    - For `timeseries` mode, raw numerical outputs are printed.
  - `--save-model`: Save the trained model after training.
  - `--model-save-path`: Path to save (or load) the trained model (default: `trained_model.pt`).
  - `--load-model`: Load a previously saved model instead of training from scratch.

### Examples

#### 1. Run on the Mackey Glass Timeseries Dataset

If you have the Mackey Glass time series file (e.g., `MackeyGlass_t17.txt` in the `./data/` folder):

```bash
python src/sparse_reservoir.py --dataset timeseries --data-file ./data/MackeyGlass_t17.txt --print-samples
```

This command runs the reservoir computing experiment on the Mackey Glass numerical data and prints a few sample numeric outputs.

#### 2. Run on a Text Dataset (Wikipedia)

The script will automatically download a Wikipedia sample (if needed), compute text embeddings using a pretrained model, and run the reservoir computing experiment. For example:

```bash
python src/sparse_reservoir.py --dataset wikipedia --data-file ./data/wikipedia_sample.txt --pretrained-model distilbert-base-uncased --print-samples
```

This command:
- Downloads and caches the Wikipedia sample if not already present.
- Computes and caches text embeddings.
- Initializes and caches reservoir weights.
- Trains the readout model.
- Generates outputs and prints a few sample decoded texts using nearest-neighbor decoding.

#### 3. Large-Scale Reservoir Experiment with Model Saving

To run a large-scale reservoir experiment (e.g., 100,000 nodes) and save the trained model for later use:

```bash
python src/sparse_reservoir.py --dataset wikipedia --data-file ./data/wikipedia_sample.txt --pretrained-model distilbert-base-uncased --print-samples --save-model --dim-res 100000 --rho 0.001 --rest
```

This command uses a large reservoir, a lower density (`rho = 0.001`), enables spectral radius estimation, saves the model to `trained_model.pt`, and prints sample outputs.

#### 4. Reload a Saved Model

To load a previously saved model (ensuring that the readout type matches the one used during saving) and generate outputs without retraining:

```bash
python src/sparse_reservoir.py --dataset wikipedia --data-file ./data/wikipedia_sample.txt --pretrained-model distilbert-base-uncased --print-samples --load-model --dim-res 100000 --rho 0.001 --rest
```

**Important:** When using `--load-model`, the model’s readout type (determined during the save) is automatically used. Ensure that you do not change the readout architecture between saving and loading.

## How Caching Works

- **Text Embeddings Caching**:  
  When processing a text dataset, the script computes embeddings using the specified pretrained model and saves the tensor to a file (e.g., `wikipedia_sample.txt.embeddings.pt`). On subsequent runs, if the cache file exists, the embeddings are loaded directly.

- **Reservoir Weights Caching**:  
  Reservoir weights (Win and W) are computed based on the reservoir parameters and saved to a file under the `cache/` directory. These are loaded in future runs when the parameters remain the same.

## Model Saving and Loading

- **Saving a Model**:  
  Use the `--save-model` flag to save the trained readout model after training to the path specified by `--model-save-path` (default: `trained_model.pt`).

- **Loading a Model**:  
  Use the `--load-model` flag to load a previously saved model. The saved model includes an extra field indicating the readout type, ensuring that the correct model architecture is instantiated on load.

## Troubleshooting

- **State Dictionary Mismatch**:  
  If you encounter errors during model loading, verify that the readout architecture used when saving the model is the same as the one when loading. The updated code saves the readout type so that the correct model is instantiated on load.

- **Sparse Tensor Warnings**:  
  PyTorch's sparse CSR tensor support is in beta. If you encounter issues, please consider filing a feature request with PyTorch.

- **Download Timeouts**:  
  If downloads from Hugging Face time out, increase the timeout by setting:
  ```bash
  export HF_HUB_DOWNLOAD_TIMEOUT=300
  ```

- **Shape Mismatch Errors**:  
  Ensure that the input/output dimensions are set correctly. For text datasets, these are automatically adjusted to the pretrained model's embedding size.

## License

[Insert your license information here.]

## Acknowledgements

- Reservoir computing concepts and implementations.
- Mackey Glass dataset for timeseries experiments.
- Hugging Face Transformers and Datasets libraries.
- PyTorch and its ecosystem.
