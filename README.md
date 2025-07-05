# Russian Voice Speech Analysis - MFCC + LSTM

A deep learning system for analyzing disordered speech patterns using MFCC feature extraction and LSTM neural networks, specifically designed for Russian voice datasets. This project is designed to run entirely in Google Colab with no local setup required.

## Author

**Email**: mohamed.ashraf.y.s.m@gmail.com

## Overview

This project implements a comprehensive speech disorder classification system that processes Russian voice audio files to detect various speech patterns including normal speech, stuttering, dysarthria, and aphasia. The system uses Mel Frequency Cepstral Coefficients (MFCC) for feature extraction and Long Short-Term Memory (LSTM) networks for classification.

**Platform**: Google Colab (No local installation required)

## Features

- **Google Colab Ready**: Complete project runs in Google Colab with zero setup
- **Automatic Installation**: All dependencies installed via pip commands in notebook
- **MFCC Feature Extraction**: Extracts 13 MFCC coefficients from audio files
- **LSTM Neural Network**: Deep learning model with multiple LSTM layers for sequence processing
- **Russian Voice Dataset Support**: Specifically designed for Russian voice patterns
- **Web Interface**: Gradio-based web application with public URL for easy audio file testing
- **Multiple Audio Formats**: Supports WAV, MP3, FLAC, M4A, and OGG files
- **Comprehensive Evaluation**: Detailed model performance metrics and visualization
- **No Local Storage**: All files saved in Colab's temporary storage (/content/)

## Dataset

The system uses the Russian Voice Dataset from Kaggle:
- **Dataset ID**: `mhantor/russian-voice-dataset`
- **Classes**: Normal speech, Stuttering, Dysarthria, Aphasia
- **Automatic Download**: Dataset automatically downloaded to `/content/` in Colab
- **Fallback**: Synthetic dataset generation if the original dataset is unavailable

### Colab Storage Locations
- **Dataset**: `/content/kagglehub/datasets/mhantor/russian-voice-dataset/`
- **Models**: `/content/models/`
- **Synthetic Data**: `/content/sample_dataset/` (if needed)

## Technical Specifications

- **Sample Rate**: 22,050 Hz
- **MFCC Coefficients**: 13
- **Sequence Length**: 174 (padded/truncated)
- **Audio Duration**: Limited to 10 seconds per file
- **Model Architecture**: Multi-layer LSTM with BatchNormalization and Dropout

## Quick Start (Google Colab)

1. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com)
2. **Create New Notebook**: Click "New notebook" or upload the project file
3. **Run All Cells**: The notebook will automatically install all dependencies and run the complete pipeline
4. **Use Web Interface**: After training, a Gradio web interface will launch with a public URL for testing

### One-Click Setup
Simply copy and paste the complete code into a Google Colab cell and run it. All dependencies will be automatically installed using pip commands included in the notebook.

```python
# All installations are handled automatically in the notebook
!pip install librosa tensorflow scikit-learn matplotlib seaborn pandas numpy gradio soundfile kagglehub
```

## Usage

### 1. Dataset Loading (Automatic in Colab)

The system automatically downloads the Russian Voice Dataset from Kaggle:

```python
import kagglehub
# This runs automatically in the notebook
dataset_path = kagglehub.dataset_download("mhantor/russian-voice-dataset")
# Dataset downloaded to /content/ directory
```

### 2. Feature Extraction

MFCC features are extracted using the following configuration:

```python
def extract_mfcc_features(file_path, sample_rate=22050, n_mfcc=13, max_pad_len=174):
    audio, sr = librosa.load(file_path, sr=sample_rate, duration=10.0)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    # Padding and normalization applied
    return mfcc
```

### 3. Model Training (Automatic)

The LSTM model is built and trained automatically:

```python
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=input_shape),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(64, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
```

### 4. Web Interface (Launches Automatically)

The Gradio web interface launches automatically with a public URL:

```python
iface = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Audio(type="filepath", label="Upload Audio File"),
    outputs=[
        gr.Textbox(label="Prediction Results"),
        gr.BarPlot(x="Class", y="Probability (%)")
    ],
    title="Russian Voice Speech Analysis System"
)
iface.launch(share=True)  # Creates public URL
```

## Model Architecture

### LSTM Network
- **Layer 1**: 128 LSTM units with return_sequences=True
- **Layer 2**: 64 LSTM units with return_sequences=True  
- **Layer 3**: 32 LSTM units with return_sequences=False
- **Dense Layers**: 64 and 32 units with ReLU activation
- **Output Layer**: Softmax activation for multi-class classification

### Regularization
- **BatchNormalization**: Applied after each LSTM and Dense layer
- **Dropout**: Rates of 0.3-0.4 to prevent overfitting
- **Early Stopping**: Monitors validation loss with patience=15
- **Learning Rate Reduction**: Reduces LR by factor of 0.5 when validation loss plateaus

## Data Preprocessing

### Feature Normalization
- **StandardScaler**: Fitted on training data and applied to all sets
- **Sequence Padding**: Fixed length of 174 time steps
- **Audio Duration**: Limited to 10 seconds per file

### Dataset Splitting
- **Training**: 60% of total data
- **Validation**: 20% of total data
- **Testing**: 20% of total data
- **Stratified Split**: Maintains class distribution across splits

## Performance Metrics

The system provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted average precision across classes
- **Recall**: Weighted average recall across classes
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown
- **Class-wise Performance**: Individual class performance metrics

## Key Improvements

### Fixed Version Features
- **Proper StandardScaler Implementation**: Correctly fitted and saved for inference
- **Enhanced Error Handling**: Robust error handling for audio processing
- **Improved Dataset Loading**: Better label extraction with fallback options
- **Gradio Interface Fixes**: Corrected output format for BarPlot visualization
- **Synthetic Dataset Generation**: Automatic fallback when real dataset unavailable

## Usage Examples

### Running in Google Colab
1. Copy the entire code into a single Colab cell
2. Run the cell - everything happens automatically
3. Wait for training to complete
4. Use the generated public URL to test audio files

### Single File Prediction (After Training)
```python
predicted_class, confidence, prob_dict = predict_speech_disorder(
    model, "path/to/audio.wav", scaler, label_encoder, class_names
)
```

### Batch Processing (After Training)
```python
for audio_file in audio_files:
    prediction = predict_speech_disorder(model, audio_file, scaler, label_encoder, class_names)
    print(f"File: {audio_file}, Prediction: {prediction}")
```

### Downloading Results from Colab
```python
# Download trained model
from google.colab import files
files.download('/content/models/russian_voice_lstm_model.h5')

# Download preprocessing objects
files.download('/content/models/preprocessing_objects.pkl')
```

## Requirements

**Google Colab Environment** (No local installation needed)
- Python 3.7+ (Pre-installed in Colab)
- GPU Runtime (Recommended for faster training)

**Dependencies** (Auto-installed via pip in notebook):
- TensorFlow 2.x
- Librosa
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Gradio
- SoundFile
- KaggleHub

### Colab Setup Tips
1. **Enable GPU**: Runtime → Change runtime type → Hardware accelerator → GPU
2. **High RAM**: Runtime → Change runtime type → Runtime shape → High-RAM
3. **Keep Session Active**: Run a cell every few hours to prevent disconnection

## License

This project is open source and available under the MIT License.

## Contributing

1. Fork the repository
2. Create a new Google Colab notebook
3. Make your changes to the code
4. Test in Colab environment
5. Submit a pull request with the updated notebook

## Important Notes for Colab Users

- **Session Limits**: Free Colab sessions have time limits. Premium users get longer sessions.
- **Storage**: All files are temporary. Download important results before session ends.
- **GPU Usage**: Training is faster with GPU enabled (Runtime → Change runtime type → GPU).
- **Public URLs**: Gradio creates temporary public URLs that expire when session ends.
- **Memory Management**: Large datasets may require High-RAM runtime configuration.

## Contact

For questions or support, contact: mohamed.ashraf.y.s.m@gmail.com

## Acknowledgments

- Russian Voice Dataset contributors
- Kaggle community for dataset hosting
- TensorFlow and Librosa development teams
- Gradio team for the web interface framework
