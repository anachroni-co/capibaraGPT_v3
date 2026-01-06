# CapibaraGPT-v2 Specialized Datasets

Comprehensive collection of specialized datasets organized by domain for training CapibaraGPT-v2.

## Dataset Categories

### 📚 **Academic** (`academic/`)
Research and educational datasets from academic institutions.

- **`wiki_datasets.py`**: Wikipedia data in multiple languages
- **`psychology_datasets.py`**: Psychology and mental health research data
- **`institutional_datasets.py`**: University and research institution datasets
- **`academic_code_datasets.py`**: Code and programming datasets from academia

```python
from capibara.data.capibara_datasets.academic import wiki_datasets

# Load Wikipedia articles in Spanish
wiki_loader = wiki_datasets.WikipediaDataset(language='es', max_articles=10000)
articles = wiki_loader.load_articles(categories=['ciencia', 'tecnología'])
```

### 💰 **Economics** (`economics/`)
Economic data, financial markets, and business intelligence.

### 🔧 **Engineering Design** (`engineering_design/`)
Engineering and technical design datasets.

- **`electronics_datasets.py`**: Electronic circuit and component data
- **`fpga_datasets.py`**: FPGA design and hardware description data

### 🧬 **Genomic** (`genomic/`)
Bioinformatics and genomic analysis datasets.

```python
from capibara.data.capibara_datasets.genomic import genomic_datasets

# Load DNA sequences for training
genomic_loader = genomic_datasets.get_genomic_loader()
dna_data = genomic_loader.load_dna_sequences(organism='human', max_sequences=1000)

# Create training batch
batch = genomic_loader.create_training_batch('dna_sequences', batch_size=32)
```

### 🔬 **Google Research** (`google_research/`)
Datasets from Google Research initiatives.

- **`google_patents_datasets.py`**: Google Patents dataset with technical specifications

### 📖 **Historical** (`historical/`)
Historical documents, archives, and cultural heritage data.

### 😄 **Humor** (`humor/`)
Comedy and humor datasets in Spanish.

```python
from capibara.data.capibara_datasets.humor import spanish_jokes

# Load Spanish jokes dataset
jokes_manager = spanish_jokes.SpanishJokesDataset()
combined_jokes = jokes_manager.get_combined_dataset()

# Filter by humor type
humor_negro = jokes_manager.filter_by_type(combined_jokes, 'humor_negro')
```

### ⚖️ **Legal** (`legal/`)
Legal documents, case law, and jurisprudence.

- International court decisions
- Trade law and disputes
- Legal document analysis

### 🔢 **Mathematics** (`mathematics/`)
Mathematical problems, proofs, and computational mathematics.

### 🎯 **Multimodal** (`multimodal/`)
Multi-modal datasets combining text, audio, video, and images.

- **`emotional_audio_datasets.py`**: Audio data with emotional annotations

### ⚡ **Physics** (`physics/`)
Physics simulations, experiments, and theoretical physics data.

### 🤖 **Robotics** (`robotics/`)
Robotics datasets for control, perception, and manipulation.

- **`advanced_robotics_datasets.py`**: Advanced robotics control data
- **`robotics_premium_datasets.py`**: Premium robotics datasets

### 🇪🇸 **Spanish Community** (`spanish_community/`)
Spanish-language community datasets.

- **`somos_nlp_datasets.py`**: SomosNLP community datasets

### 🏛️ **Spanish Government** (`spanish_government/`)
Official Spanish government datasets and documents.

### 🔬 **Specialized Research** (`specialized_research/`)
Specialized research datasets from various domains.

- **`archaeology_datasets.py`**: Archaeological data and cultural heritage

### 💻 **Systems** (`systems/`)
System administration, DevOps, and infrastructure datasets.

- **`linux_datasets.py`**: Linux system data and configurations  
- **`systems_logs_datasets.py`**: System logs and monitoring data

### 👁️ **Vision** (`vision/`)
Computer vision and image processing datasets.

## Key Features

### 🚀 **High Performance**
- **JAX Integration**: Optimized for TPU/GPU acceleration
- **Streaming Support**: Efficient memory usage for large datasets
- **Batch Processing**: Configurable batch sizes and sampling strategies

### 🌐 **Multi-language Support**
- **Spanish Focus**: Extensive Spanish-language datasets
- **International**: Global datasets in multiple languages
- **Cultural Context**: Culturally-aware data processing

### 🔐 **Premium Access**
- **Authentication**: Secure access to premium datasets
- **Rate Limiting**: Respectful API usage
- **Compliance**: GDPR, HIPAA, and other regulatory compliance

### 📊 **Rich Metadata**
- **Dataset Statistics**: Comprehensive statistics and analysis
- **Quality Metrics**: Data quality assessment
- **Documentation**: Detailed documentation for each dataset

## Usage Examples

### Loading Multiple Domain Datasets
```python
from capibara.data.capibara_datasets import (
    academic, legal, humor, genomic
)

# Create multi-domain training data
datasets = {
    'academic': academic.wiki_datasets.load_wikipedia('es'),
    'legal': legal.legal_datasets.load_icj_cases(),
    'humor': humor.spanish_jokes.load_chistes_spanish_jokes(),
    'genomic': genomic.genomic_datasets.get_genomic_loader().load_dna_sequences()
}
```

### Advanced Dataset Mixing
```python
from capibara.data.core import MultiDatasetLoader
from capibara.data.capibara_datasets import humor, academic

# Mix humor and academic datasets
humor_ds = humor.spanish_jokes.SpanishJokesDataset().get_combined_dataset()
academic_ds = academic.wiki_datasets.WikipediaDataset().load_articles()

# Configure weighted sampling
multi_loader = MultiDatasetLoader(
    datasets=[humor_ds, academic_ds],
    weights=[0.3, 0.7],  # 30% humor, 70% academic
    batch_sizes=[16, 32]
)

for batch in multi_loader:
    # Process mixed batch
    train_model(batch)
```

### Domain-Specific Processing
```python
from capibara.data.capibara_datasets.genomic import genomic_datasets

# Genomic data with specialized processing
genomic_loader = genomic_datasets.GenomicDatasetLoader()

# Load and encode DNA sequences
dna_data = genomic_loader.load_dna_sequences(organism='human')
for sequence in dna_data['sequences']:
    encoded = sequence['encoded']  # Numerical encoding
    one_hot = genomic_loader.processor.one_hot_encode_dna(sequence['sequence'])
```

## Dataset Statistics

- **Total Domains**: 17 specialized categories
- **Languages**: Primarily Spanish, with multilingual support
- **Size Range**: From small specialized sets (~500 samples) to massive corpora (1M+ samples)
- **Formats**: JSON, CSV, FASTA, XML, Parquet, PDF, and more
- **Access Levels**: Public, API-based, institutional, and premium datasets

## Integration

Seamlessly integrates with:
- **capibara.data.core**: Core data processing infrastructure  
- **capibara.data.configs**: Dataset access and configuration management
- **capibara.training**: Model training pipelines
- **capibara.jax**: JAX/TPU optimization
- **capibara.layers**: Specialized model layers for domain-specific data