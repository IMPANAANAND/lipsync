# LipSync Real vs Fake Detection

This project uses deep learning to classify videos as "real" or "fake" based on both video frames and audio features (MFCCs). It is designed for research and experimentation with multimodal deepfake detection.

## Features

- Loads video frames and audio MFCC features for each sample.
- Trains a multimodal neural network using TensorFlow/Keras.
- Supports batch processing for efficient training.
- Includes scripts for prediction on new videos (with or without pre-extracted frames/audio).
- Handles both real and fake video/audio datasets.

## Requirements

- Python 3.7+
- TensorFlow / Keras
- OpenCV
- NumPy
- librosa
- scikit-learn
- tqdm
- matplotlib
- moviepy

Install dependencies with:

```sh
pip install tensorflow opencv-python numpy librosa scikit-learn tqdm matplotlib moviepy
```

## Project Structure

- `accuracy lipsync.ipynb` — Main notebook with all code for data loading, model training, and prediction.
- Video and audio data folders (update paths in the notebook as needed).

## Usage

### Training

1. Place your real and fake video/audio data in the specified folders.
2. Adjust the paths in the notebook to match your data locations.
3. Run the notebook cells to train the model.
4. The trained model will be saved as `abcde.h5`.

### Prediction

- Use the provided prediction functions to classify new videos as real or fake.
- You can predict using either pre-extracted frames/audio or directly from a video file.

## Example

```python
# Predict from a video file
result = predict_real_or_fake("path/to/video.mp4", img_size=224, num_frames=10)
print(f"The given video and audio are: {result}")
```

## Notes

- Make sure your data is organized as expected (see the notebook for folder structure).
- Adjust hyperparameters (image size, batch size, epochs, etc.) as needed for your hardware.

## License

For research and educational purposes only.
