# PLMEmbedder

**Protein Language Model Embedding Library**

PLMEmbedder is a standalone library for computing protein sequence embeddings using state-of-the-art protein language models (PLMs). It provides a clean API and CLI for embedding proteins, caching results, and generating decoy sequences.

---

## Overview

PLMEmbedder takes protein sequences in FASTA format and:

1. **Computes PLM embeddings** for each amino acid using protein language models (ESM2, ESM1b, ProtBert, ProtT5)
2. **Caches embeddings** for reuse across multiple analyses
3. **Generates decoy sequences** using various methods (reversed, permuted, shuffled blocks)
4. **Provides a clean API** for integration with downstream analysis pipelines

---

## Installation

### From PyPI (recommended)

```bash
pip install plmembedder
```

### From source

```bash
git clone https://github.com/biodataganache/plmembedder.git
cd plmembedder
pip install -e .
```

### With specific PLM support

```bash
# ESM models only
pip install plmembedder[esm]

# ProtBert models only
pip install plmembedder[protbert]

# ProtT5 models only
pip install plmembedder[prott5]

# All PLM support
pip install plmembedder[all]
```

---

## System Requirements

### Minimum Requirements
- Python 3.8 or higher
- 8 GB RAM
- 10 GB free disk space

### Recommended Requirements
- Python 3.9 or 3.10
- 16+ GB RAM
- NVIDIA GPU with 8+ GB VRAM (for GPU acceleration)
- CUDA 11.0+ and cuDNN 8.0+
- 20+ GB free disk space

---

## Quick Start

### Command Line

```bash
# Basic embedding
plmembedder proteins.fasta -o results/

# With caching (recommended for large datasets)
plmembedder proteins.fasta --cache-embeddings -c cache/ -o results/

# Use cached embeddings for subsequent runs
plmembedder proteins.fasta --cache-embeddings -c cache/ -o results2/

# Generate decoy sequences
plmembedder proteins.fasta --n-decoys 1 --decoy-type reversed -o results/

# Use a smaller/faster model
plmembedder proteins.fasta --model esm2_t6_8M_UR50D -o results/

# CPU only (no GPU)
plmembedder proteins.fasta --device cpu -o results/
```

### Python API

```python
from plmembedder import EmbeddingPipeline, PipelineConfig, CacheConfig

# Simple usage with defaults
pipeline = EmbeddingPipeline()
embeddings = pipeline.run("proteins.fasta")

# With caching enabled
config = PipelineConfig(
    cache=CacheConfig(enabled=True, cache_dir="cache/")
)
pipeline = EmbeddingPipeline(config)
embeddings = pipeline.run("proteins.fasta", output_dir="results/")

# Step-by-step for more control
pipeline = EmbeddingPipeline(config)
pipeline.load_sequences("proteins.fasta")
pipeline.generate_decoys()  # If configured
pipeline.compute_embeddings()

# Access individual embeddings
for seq_id, emb_result in pipeline.iterate_embeddings():
    print(f"{seq_id}: shape {emb_result.embeddings.shape}")
    # emb_result.embeddings is shape (seq_len, embedding_dim)
```

---

## CLI Reference

```
usage: plmembedder [-h] [-o OUTPUT] [--model MODEL] [--model-type {esm2,esm1b,protbert,prott5}]
                   [--device DEVICE] [--batch-size BATCH_SIZE] [--layer LAYER]
                   [--max-length MAX_LENGTH] [--cache-embeddings] [-c CACHE_DIR]
                   [--n-decoys N_DECOYS] [--decoy-type {reversed,permuted,shuffled_blocks}]
                   [--decoy-prefix DECOY_PREFIX] [--decoy-seed DECOY_SEED]
                   [--decoys-only] [--write-decoy-fasta WRITE_DECOY_FASTA]
                   [--max-sequences MAX_SEQUENCES] [--no-validate] [-v] [--embed-only]
                   fasta

Protein Language Model Embedding Pipeline

positional arguments:
  fasta                 Input FASTA file

options:
  -h, --help            show this help message and exit
  -o, --output OUTPUT   Output directory (default: output)
  -v, --verbose         Verbose output

Model Parameters:
  --model MODEL         Model name (default: esm2_t33_650M_UR50D)
  --model-type          Model type: esm2, esm1b, protbert, prott5 (default: esm2)
  --device DEVICE       Device: cuda or cpu (default: cuda)
  --batch-size          Batch size for embedding (default: 4)
  --layer LAYER         Model layer to extract from (default: -1, last layer)
  --max-length          Maximum sequence length (default: 1024)

Caching Parameters:
  --cache-embeddings    Enable embedding caching
  -c, --cache-dir       Cache directory (default: embeddings_cache)

Decoy Parameters:
  --n-decoys N_DECOYS   Number of decoys per sequence (default: 0)
  --decoy-type          Decoy method: reversed, permuted, shuffled_blocks
  --decoy-prefix        Prefix for decoy IDs (default: DECOY_)
  --decoy-seed          Random seed for decoy generation
  --decoys-only         Embed only decoy sequences
  --write-decoy-fasta   Write decoys to FASTA file

Input Options:
  --max-sequences       Maximum sequences to process
  --no-validate         Skip sequence validation

Output Options:
  --embed-only          Only compute embeddings, skip saving consolidated output
```

---

## Supported Models

### ESM2 (Recommended)
| Model | Parameters | Embedding Dim | Memory |
|-------|------------|---------------|--------|
| `esm2_t33_650M_UR50D` | 650M | 1280 | ~3GB |
| `esm2_t30_150M_UR50D` | 150M | 640 | ~1GB |
| `esm2_t12_35M_UR50D` | 35M | 480 | ~500MB |
| `esm2_t6_8M_UR50D` | 8M | 320 | ~200MB |

### ESM1b
| Model | Parameters | Embedding Dim |
|-------|------------|---------------|
| `esm1b_t33_650M_UR50S` | 650M | 1280 |

### ProtBert
| Model | Embedding Dim |
|-------|---------------|
| `Rostlab/prot_bert` | 1024 |
| `Rostlab/prot_bert_bfd` | 1024 |

### ProtT5
| Model | Embedding Dim |
|-------|---------------|
| `Rostlab/prot_t5_xl_half_uniref50-enc` | 1024 |
| `Rostlab/prot_t5_xl_uniref50` | 1024 |
| `Rostlab/prot_t5_base_mt_uniref50` | 768 |

---

## API Reference

### Core Classes

#### `EmbeddingPipeline`

Main pipeline for computing protein embeddings.

```python
from plmembedder import EmbeddingPipeline, PipelineConfig

pipeline = EmbeddingPipeline(config: PipelineConfig = None)

# Methods
pipeline.load_sequences(fasta_path: str) -> List[Tuple[str, str]]
pipeline.generate_decoys() -> List[DecoyResult]
pipeline.compute_embeddings() -> Dict[str, EmbeddingResult]
pipeline.run(fasta_path: str, output_dir: str = None) -> Dict[str, EmbeddingResult]
pipeline.get_embedding(sequence_id: str) -> EmbeddingResult
pipeline.iterate_embeddings() -> Iterator[Tuple[str, EmbeddingResult]]
pipeline.save_embeddings(output_dir: str) -> None
```

#### `EmbeddingResult`

Container for embedding results.

```python
from plmembedder import EmbeddingResult

result.sequence_id: str           # Sequence identifier
result.sequence: str              # Amino acid sequence
result.embeddings: np.ndarray     # Shape: (seq_length, embedding_dim)
result.attention_weights: np.ndarray  # Optional attention weights
```

#### `EmbeddingCache`

Manage cached embeddings.

```python
from plmembedder import EmbeddingCache

cache = EmbeddingCache(cache_dir: str, embedder_config: EmbedderConfig)
cache.get(sequence_id: str, sequence: str) -> Optional[EmbeddingResult]
cache.save(result: EmbeddingResult) -> None
cache.clear() -> None
```

#### `DecoyGenerator`

Generate decoy sequences.

```python
from plmembedder import DecoyGenerator, DecoyConfig, DecoyType

config = DecoyConfig(
    n_decoys=1,
    decoy_type=DecoyType.REVERSED,
    decoy_prefix="DECOY_"
)
generator = DecoyGenerator(config)
decoys = generator.generate_all(sequences: List[Tuple[str, str]])
```

### Configuration Classes

```python
from plmembedder import (
    PipelineConfig,
    EmbedderConfig,
    CacheConfig,
    DecoyConfig,
    OutputConfig,
    EmbedderType,
    DecoyType,
)

# Full configuration example
config = PipelineConfig(
    embedder=EmbedderConfig(
        embedder_type=EmbedderType.ESM2,
        model_name="esm2_t33_650M_UR50D",
        device="cuda",
        batch_size=4,
        layer=-1,
        max_sequence_length=1024,
    ),
    cache=CacheConfig(
        enabled=True,
        cache_dir="embeddings_cache",
    ),
    decoy=DecoyConfig(
        n_decoys=1,
        decoy_type=DecoyType.REVERSED,
        decoy_prefix="DECOY_",
        random_seed=42,
        decoys_only=False,
    ),
    output=OutputConfig(
        output_dir="output",
        save_embeddings=True,
        save_format="npz",
    ),
    max_sequences=None,
    validate_sequences=True,
)
```

---

## Integration with snaikmer

PLMEmbedder is designed to work seamlessly with [snaikmer](https://github.com/biodataganache/snaikmer) for k-mer embedding analysis:

```python
from plmembedder import EmbeddingPipeline, PipelineConfig
from snaikmer import KmerEmbeddingPipeline

# Step 1: Compute embeddings with plmembedder
embed_config = PipelineConfig(
    cache=CacheConfig(enabled=True, cache_dir="cache/")
)
embed_pipeline = EmbeddingPipeline(embed_config)
embed_pipeline.load_sequences("proteins.fasta")
sequence_embeddings = embed_pipeline.compute_embeddings()

# Step 2: Pass to snaikmer for k-mer analysis
kmer_pipeline = KmerEmbeddingPipeline(kmer_config)
kmer_pipeline.analyze_with_embeddings(
    sequences=embed_pipeline.sequences,
    embeddings=sequence_embeddings
)
```

---

## HPC / Batch Processing

For large-scale embedding on HPC systems:

```bash
# Pre-compute and cache embeddings (no downstream analysis)
plmembedder proteins.fasta --embed-only --cache-embeddings -c /scratch/cache/

# Generate decoys and write to FASTA for external tools
plmembedder proteins.fasta --n-decoys 1 --write-decoy-fasta decoys.fasta --embed-only
```

---

## Output Format

Embeddings are saved in NumPy's compressed `.npz` format:

```
output/
├── embeddings.npz    # All embeddings (sequence_id -> embedding array)
└── metadata.json     # Sequence metadata and shapes
```

Loading saved embeddings:

```python
import numpy as np
import json

# Load embeddings
data = np.load("output/embeddings.npz")
for seq_id in data.files:
    embedding = data[seq_id]  # Shape: (seq_len, embedding_dim)

# Load metadata
with open("output/metadata.json") as f:
    metadata = json.load(f)
```

---

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch-size 1`
- Use a smaller model: `--model esm2_t6_8M_UR50D`
- Use CPU: `--device cpu`

### Model Download Issues
Models are downloaded automatically on first use. To pre-download:

```python
# ESM2
import esm
esm.pretrained.esm2_t33_650M_UR50D()

# ProtBert/ProtT5
from transformers import AutoModel
AutoModel.from_pretrained("Rostlab/prot_bert")
```

### Invalid Sequences
By default, invalid amino acid characters are removed. To skip validation:
```bash
plmembedder proteins.fasta --no-validate -o results/
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

## Citation

If you use PLMEmbedder in your research, please cite the underlying models:

- **ESM2**: Lin et al. (2022). "Language models of protein sequences at the scale of evolution enable accurate structure prediction."
- **ProtBert/ProtT5**: Elnaggar et al. (2021). "ProtTrans: Towards Cracking the Language of Life's Code Through Self-Supervised Deep Learning and High Performance Computing."
```

This README provides comprehensive documentation for the plmembedder library, including installation instructions, CLI usage, Python API reference, supported models, and integration guidance with snaikmer.
