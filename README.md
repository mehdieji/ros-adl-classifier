# ROS ADL Classifier

A comprehensive ROS-based system for Activities of Daily Living (ADL) classification using machine learning and sensor data.

## Overview

This project implements a complete pipeline for real-time classification of human activities using sensor data (IMU, camera) in a ROS environment. The system is designed for assistive robotics and health monitoring applications.

## Features

- **Modular Architecture**: Separate modules for preprocessing, segmentation, feature extraction, and classification
- **Real-time Processing**: ROS integration for live data processing and classification
- **Flexible Configuration**: YAML-based configuration system
- **Multiple ML Models**: Support for various classification algorithms
- **Comprehensive Feature Extraction**: Time-domain, frequency-domain, and statistical features
- **Data Pipeline**: Complete pipeline from raw sensor data to activity predictions

## Project Structure

```
ros-adl-classifier/
├── src/                          # Source code modules
│   ├── data_preprocessing/       # Data cleaning and filtering
│   ├── segmentation/            # Time series windowing
│   ├── feature_extraction/      # Feature engineering
│   ├── models/                  # ML models and training
│   └── realtime/               # ROS nodes and real-time processing
├── data/                        # Data directories (gitignored)
│   ├── raw/                    # Raw sensor data
│   ├── processed/              # Cleaned and filtered data
│   ├── features/               # Extracted features
│   └── models/                 # Trained models
├── ros/                         # ROS-specific files
│   ├── launch/                 # Launch files
│   ├── msg/                    # Custom messages
│   ├── srv/                    # Service definitions
│   └── param/                  # Parameter files
├── config/                      # Configuration files
├── scripts/                     # Executable scripts
├── notebooks/                   # Jupyter notebooks for analysis
├── tests/                       # Unit tests
├── docs/                        # Documentation
├── results/                     # Experiment results
└── logs/                        # Log files
```

## Installation

### Prerequisites

- ROS (tested with ROS Noetic)
- Python 3.10+ (recommended)
- Conda/Mamba (recommended for environment management)
- Required Python packages (see `requirements.txt`)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ros-adl-classifier
```

2. Create and activate a virtual environment:
```bash
# Using mamba (recommended)
mamba create -n ros-adl-classifier python=3.10
mamba activate ros-adl-classifier

# Alternative: using conda
conda create -n ros-adl-classifier python=3.10
conda activate ros-adl-classifier
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Build ROS package:
```bash
catkin_make
source devel/setup.bash
```

## Configuration

Edit `config.yaml` to customize:
- Sensor parameters
- Preprocessing settings
- Feature extraction options
- Model parameters
- ROS topics and node settings

## Usage

### Training a Model

```bash
python scripts/train_model.py --data-path /path/to/training/data
```

### Real-time Classification

Launch the ROS system:
```bash
roslaunch ros-adl-classifier adl_classifier.launch
```

### Data Processing Pipeline

```python
import sys
from pathlib import Path
import yaml

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_preprocessing import DataPreprocessor
from segmentation import WindowSegmenter
from feature_extraction import FeatureExtractor
from models import ADLClassifier

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize pipeline
preprocessor = DataPreprocessor(config)
segmenter = WindowSegmenter(config)
feature_extractor = FeatureExtractor(config)
classifier = ADLClassifier(config)

# Process data
processed_data = preprocessor.preprocess(raw_data)
segments = segmenter.segment(processed_data)
features = feature_extractor.extract_features(segments)
predictions = classifier.predict(features)
```

## Supported Activities

The system can classify the following activities:
- Walking
- Sitting
- Standing
- Lying down
- Eating
- Drinking
- Cooking
- Cleaning

## Development

### Adding New Features

1. Create feature extraction functions in `src/feature_extraction/`
2. Update the configuration in `config.yaml`
3. Add tests in `tests/`

### Adding New Models

1. Implement model class in `src/models/`
2. Follow the `BaseClassifier` interface
3. Update configuration options

### Running Tests

```bash
pytest tests/
```

### Updating requirements.txt

To update your `requirements.txt` with only pip-installed (PyPI) packages, run:
```bash
pip freeze | grep -v '@ file://' > requirements.txt
```
This will exclude packages installed from local files or conda build artifacts.

## API Reference

See the `docs/` directory for detailed API documentation.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{ros_adl_classifier,
  title={ROS ADL Classifier},
  author={Your Name},
  year={2024},
  url={https://github.com/mehdieji/ros-adl-classifier}
}
```

## Acknowledgments

- SCAI-Lab, Swiss Paraplegic Research, and ETH Zurich
- Contributors to the open-source ML libraries used in this project

## Generating ADL Event Plots for a Patient

You can automatically generate a multi-page PDF with synchronized plots for all ADL events and sensor modalities for a given patient using the provided script.

### Configuration

Specify the patient ID an time tick intervals in `config/plot_config.yaml`:

```yaml
patient_id: "01"
time_tick_interval: 2
```

### Usage

Run the following command from the project root:

```bash
python scripts/generate_adl_event_plots.py
```

- The script will parse all available ADL event data for the specified patient and generate a PDF in the `results/` directory (e.g., `results/patient_01_adl_events.pdf`).
- Each page of the PDF corresponds to one event instance, with all sensor plots for that event.
- The PDF is automatically excluded from git (see `.gitignore`).
