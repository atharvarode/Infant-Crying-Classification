import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os


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
# input_audio_file = 'C:/Users/kalli/Desktop/demo.wav'
# output_image_file = 'output_spectrogram1.png'
# audio_to_spectrogram(input_audio_file, output_image_file)

# # Function to convert audio files in a folder to spectrograms
# def convert_folder_to_spectrograms(folder_path):
#     # List all audio files in the folder
#     audio_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

#     # Iterate through each audio file
#     for audio_file in audio_files:
#         # Construct full path to the audio file
#         audio_file_path = os.path.join(folder_path, audio_file)
        
#         # Define output image path
#         output_image_path = os.path.join(folder_path, audio_file.replace('.wav', '_spectrogram.png'))
        
#         # Convert audio to spectrogram and save the image
#         audio_to_spectrogram(audio_file_path, output_image_path)

# # List of folders containing labeled audio data
# folders = ['belly_pain', 'discomfort', 'hungry', 'tired', 'burping']

# # Iterate through each folder
# for folder in folders:
#     folder_path = os.path.join('C:/Users/kalli/Documents/GitHub/Infant-Crying-Classification/Data/', folder)  # Update with the parent directory path
#     convert_folder_to_spectrograms(folder_path)

# Function to convert audio files in a folder to spectrograms
def convert_folder_to_spectrograms(input_folder_path, output_folder_path, label):
    # List all audio files in the folder
    audio_files = [f for f in os.listdir(input_folder_path) if f.endswith('.wav')]

    # Iterate through each audio file
    for i, audio_file in enumerate(audio_files):
        # Construct full path to the audio file
        audio_file_path = os.path.join(input_folder_path, audio_file)
        
        # Define output image path
        output_image_path = os.path.join(output_folder_path, f"{label}_{i+1}_spectrogram.png")
        
        # Convert audio to spectrogram and save the image
        audio_to_spectrogram(audio_file_path, output_image_path)

# List of folders containing labeled audio data
folders = ['belly_pain', 'discomfort', 'hungry', 'tired', 'burping']

# Define output folder path
output_parent_folder_path = 'C:/Users/kalli/Documents/GitHub/Infant-Crying-Classification/Photos/'

# Iterate through each folder
for folder in folders:
    input_folder_path = os.path.join('C:/Users/kalli/Documents/GitHub/Infant-Crying-Classification/Data/', folder)  
    output_folder_path = os.path.join(output_parent_folder_path, folder)
    os.makedirs(output_folder_path, exist_ok=True)  # Create the output folder if it doesn't exist
    convert_folder_to_spectrograms(input_folder_path, output_folder_path, folder)
