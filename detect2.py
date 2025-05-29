import gradio as gr
import os
from detect import SimpleOfflineAccentClassifier
import ssl
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

import torch
import torchaudio
import librosa
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import soundfile as sf

class AccentClassifierApp:
    def __init__(self):
        self.classifier = HuggingFaceAccentClassifier()
        
    def classify_audio(self, audio_file):
        if audio_file is None:
            return "Please upload an audio file."
        
        try:
            result = self.classifier.predict_accent(audio_file)
            
            if result is None:
                return "Audio file processing failed."
            
            output = f"Predicted Accent: {result['accent']}\n"
            output += f"Confidence Score: {result['confidence']:.2%}\n\n"
            output += "All Probabilities:\n"
            
            sorted_probs = sorted(
                result['all_probabilities'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            for accent, prob in sorted_probs:
                bar = "█" * int(prob * 20)
                output += f"- {accent}: {prob:.2%} {bar}\n"
            
            return output
            
        except Exception as e:
            return f"Error occurred: {str(e)}"
    
    def create_interface(self):
        with gr.Blocks(title="Accent Classifier") as interface:
            gr.Markdown("""
            # AI Accent Classifier
            
            This application analyzes speech audio files to predict accents.
            Supported formats: WAV, MP3, FLAC
            """)
            
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(
                        label="Upload Audio File",
                        type="filepath"
                    )
                    
                    classify_btn = gr.Button(
                        "Analyze Accent", 
                        variant="primary"
                    )
                
                with gr.Column():
                    output_text = gr.Markdown(
                        label="Analysis Results",
                        value="Analysis results will appear here..."
                    )
            
            gr.Markdown("### Example Audio Files")
            gr.Examples(
                examples=[
                    ["examples/american_sample.wav"],
                    ["examples/british_sample.wav"],
                ] if os.path.exists("examples") else [],
                inputs=audio_input
            )
            
            classify_btn.click(
                fn=self.classify_audio,
                inputs=audio_input,
                outputs=output_text
            )
        
        return interface

    def extract_acoustic_features(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=22050, duration=30)
            
            if len(y) == 0:
                return None
            
            min_length = sr * 2
            if len(y) < min_length:
                repeat_count = int(min_length / len(y)) + 1
                y = np.tile(y, repeat_count)[:min_length]
            
            features = {}
            
            n_fft = min(2048, len(y))
            hop_length = n_fft // 4
            
            try:
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
                features['mfcc_mean'] = np.mean(mfccs, axis=1)
                features['mfcc_std'] = np.std(mfccs, axis=1)
            except Exception as e:
                features['mfcc_mean'] = np.zeros(13)
                features['mfcc_std'] = np.zeros(13)
            
            try:
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
                features['spectral_centroid'] = float(np.mean(spectral_centroids))
                features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            except Exception as e:
                features['spectral_centroid'] = 1500.0
                features['spectral_centroid_std'] = 100.0
            
            try:
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1, n_fft=n_fft, hop_length=hop_length)
                pitch_values = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_values.append(pitch)
                
                if pitch_values:
                    features['pitch_mean'] = float(np.mean(pitch_values))
                    features['pitch_std'] = float(np.std(pitch_values))
                else:
                    features['pitch_mean'] = 150.0
                    features['pitch_std'] = 20.0
            except Exception as e:
                features['pitch_mean'] = 150.0
                features['pitch_std'] = 20.0
            
            try:
                zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
                features['zcr_mean'] = float(np.mean(zcr))
                features['zcr_std'] = float(np.std(zcr))
            except Exception as e:
                features['zcr_mean'] = 0.1
                features['zcr_std'] = 0.05
            
            return features
            
        except Exception as e:
            return None

def main():
    app = AccentClassifierApp()
    interface = app.create_interface()
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )

if __name__ == "__main__":
    main()