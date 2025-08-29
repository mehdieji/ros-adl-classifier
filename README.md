# ROS ADL Classifier

A comprehensive ROS-based system for Activities of Daily Living (ADL) classification using machine learning and sensor data.

## Overview

This project implements a complete pipeline for real-time classification of human activities using sensor data (IMU, camera) in a ROS environment. The system is designed for assistive robotics and health monitoring applications.

## Features

- **Modular Architecture**: Separate modules for preprocessing, segmentation, and classification
- **Real-time Processing**: ROS integration for live data processing and classification
- **Flexible Configuration**: YAML-based configuration system
- **Multiple ML Models**: Support for various classification algorithms
- **Comprehensive Feature Extraction**: Time-domain, frequency-domain, and statistical features using tsfresh
- **Data Pipeline**: Complete pipeline from raw sensor data to activity predictions
- **Batch Feature Extraction**: Standalone script for processing multiple patients and ADL events

## Project Structure

```
ros-adl-classifier/
├── src/                          # Source code modules
│   ├── data_preprocessing/       # Data cleaning and filtering
│   ├── segmentation/            # Time series windowing
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
│   └── run_feature_extraction.py # Feature extraction pipeline
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
from models import ADLClassifier

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize pipeline
preprocessor = DataPreprocessor(config)
segmenter = WindowSegmenter(config)
classifier = ADLClassifier(config)

# Process data
processed_data = preprocessor.preprocess(raw_data)
segments = segmenter.segment(processed_data)
predictions = classifier.predict(segments)
```

## Supported Activities

The system can classify the following activities:
- Resting
- Propulsion
- Lying down
- Eating
- etc.

  
## Development

### Adding New Features

1. Update the feature extraction script `scripts/run_feature_extraction.py` to include new features
2. Update the configuration in `config/feature_config.yaml`
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

## Feature Extraction Pipeline

The project includes a comprehensive feature extraction pipeline that processes sensor data from multiple patients and ADL events, extracting time-domain and statistical features using the tsfresh library.

### Overview

The feature extraction process:
1. **Data Discovery**: Automatically finds all ADL event instances for specified patients
2. **Sensor Parsing**: Parses data from M5StickC sensors (wrist left/right, wheel), Polar chest sensor, and Sensomative pressure sensor
3. **Windowing**: Segments time series data into overlapping windows
4. **Feature Extraction**: Extracts comprehensive features using tsfresh (statistical, frequency-domain, entropy-based features)
5. **Output**: Saves feature matrices as CSV files for each ADL event instance

### Configuration

Configure the feature extraction process in `config/feature_config.yaml`:

```yaml
# Patient IDs to process (extract just the number from 'patient01', 'patient02', etc.)
patient_ids:
  - "01"
  - "02"
  - "03"
  - "04"
  - "05"

# ADL events to process
adl_events:
  - "synchronization"
  - "emptywc"
  - "resting"
  - "phone"
  - "computer"
  - "arm_raises"
  - "eating"
  - "hand_cycling"
  - "chair_to_bed_transfer"
  - "bed_to_chair_transfer"
  - "pressure_relief"
  - "laying_on_back"
  - "laying_on_right"
  - "laying_on_left"
  - "laying_on_stomach"
  - "assisted_propulsion"
  - "self_propulsion"

# Feature extraction parameters
extraction_params:
  window_size: 4  # in seconds
  step_size: 2    # in seconds
  n_jobs: 20      # number of parallel jobs for feature extraction
  disable_progressbar: false

# Output settings
output:
  features_dir: "data/features"  # relative to project root
  filename_format: "patient_{patient_id}_{adl_event}_{instance_id}_features.csv"

# NaN/Inf handling for feature output
nan_inf_handling:
  nan_value: 11000      # Value to replace NaN
  posinf_value: 9000   # Value to replace +Inf
  neginf_value: -9000  # Value to replace -Inf
```

### NaN/Inf Handling

After feature extraction, all NaN, +Inf, and -Inf values in the output feature CSVs are replaced with the values specified in the `nan_inf_handling` section of the config. This ensures downstream ML code can safely process the features without encountering missing or infinite values.

### Supported Features

The pipeline extracts the following feature categories using tsfresh:
- **Statistical Features**: mean, standard deviation, variance, min/max, median, skewness, kurtosis
- **Change Features**: absolute sum of changes, mean absolute change, mean change
- **Strike Features**: longest strike above/below mean
- **Correlation Features**: autocorrelation, aggregated autocorrelation
- **Entropy Features**: binned entropy, sample entropy
- **Peak Features**: number of peaks
- **Energy Features**: absolute energy
- **Frequency Features**: FFT coefficients, spectral density
- **Crossing Features**: number of crossings

### Usage

Run the feature extraction pipeline from the project root:

```bash
python scripts/run_feature_extraction.py
```

### Output

The pipeline generates CSV files in `data/features/` with the naming convention:
```
patient_{patient_id}_{adl_event}_{instance_id}_features.csv
```

Each CSV file contains:
- **window_id**: Identifier for each time window
- **patient**: Patient ID
- **ADL_class**: ADL event type
- **Feature columns**: All extracted features with format `{sensor}|{modality}|{channel}|{feature_name}`

Example feature columns:
- `M5 Wrist L|linear_acceleration|linear_acceleration_x|value__mean`
- `Polar Chest|linear_acceleration|linear_acceleration_y|value__standard_deviation`
- `Sensomative Bottom|pressure|pressure_0|value__abs_energy`

### Processing Details

- **Multi-sensor Support**: Processes data from 5 sensors (3 M5StickC, 1 Polar, 1 Sensomative)
- **Multi-modal Features**: Extracts features from linear acceleration, angular velocity, and pressure data
- **Robust Error Handling**: Continues processing even if individual instances fail
- **Progress Tracking**: Shows detailed progress for each patient and ADL event
- **Parallel Processing**: Uses multiple CPU cores for faster feature extraction

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

