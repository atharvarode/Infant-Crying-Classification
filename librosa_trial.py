import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def audio_to_spectrogram(audio_file, output_image):
    # Load the audio file
    y, sr = librosa.load(audio_file)

    # Compute the magnitude spectrogram
    D = np.abs(librosa.stft(y))

    # Convert to dB scale
    D_db = librosa.amplitude_to_db(D, ref=np.max)

    # Plot the spectrogram
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(D_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig(output_image)
    plt.close()

# Example usage
input_audio_file = 'C:/Users/kalli/Desktop/demo.wav'
output_image_file = 'output_spectrogram1.png'
audio_to_spectrogram(input_audio_file, output_image_file)


