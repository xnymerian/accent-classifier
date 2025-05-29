import librosa
import numpy as np
import os
import sys

class SimpleOfflineAccentClassifier:
    def __init__(self):
        self.accent_profiles = {
            'American': {
                'formant_f1_range': (300, 800),
                'formant_f2_range': (1200, 2200),
                'pitch_variance': 'medium',
                'tempo_range': (140, 180),
                'spectral_tilt': 'neutral'
            },
            'British': {
                'formant_f1_range': (280, 750),
                'formant_f2_range': (1400, 2400),
                'pitch_variance': 'low',
                'tempo_range': (120, 160),
                'spectral_tilt': 'high'
            },
            'Australian': {
                'formant_f1_range': (320, 850),
                'formant_f2_range': (1100, 2000),
                'pitch_variance': 'high',
                'tempo_range': (130, 170),
                'spectral_tilt': 'low'
            },
            'Indian': {
                'formant_f1_range': (350, 900),
                'formant_f2_range': (1300, 2300),
                'pitch_variance': 'high',
                'tempo_range': (160, 200),
                'spectral_tilt': 'neutral'
            },
            'Canadian': {
                'formant_f1_range': (290, 780),
                'formant_f2_range': (1250, 2150),
                'pitch_variance': 'medium',
                'tempo_range': (135, 175),
                'spectral_tilt': 'neutral'
            }
        }
    
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
                
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
                features['spectral_rolloff'] = float(np.mean(spectral_rolloff))
                
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
                features['spectral_bandwidth'] = float(np.mean(spectral_bandwidth))
            except Exception as e:
                features['spectral_centroid'] = 1500.0
                features['spectral_centroid_std'] = 100.0
                features['spectral_rolloff'] = 3000.0
                features['spectral_bandwidth'] = 1000.0
            
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
                    features['pitch_range'] = float(np.max(pitch_values) - np.min(pitch_values))
                else:
                    features['pitch_mean'] = 150.0
                    features['pitch_std'] = 20.0
                    features['pitch_range'] = 50.0
            except Exception as e:
                features['pitch_mean'] = 150.0
                features['pitch_std'] = 20.0
                features['pitch_range'] = 50.0
            
            try:
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
                features['tempo'] = float(tempo)
            except Exception as e:
                features['tempo'] = 120.0
            
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
    
    def calculate_accent_scores(self, features):
        scores = {}
        
        for accent, profile in self.accent_profiles.items():
            score = 0.0
            
            spectral_centroid = features.get('spectral_centroid', 1500)
            f2_range = profile['formant_f2_range']
            
            if f2_range[0] <= spectral_centroid <= f2_range[1]:
                score += 0.3
            else:
                distance = min(
                    abs(spectral_centroid - f2_range[0]),
                    abs(spectral_centroid - f2_range[1])
                )
                score += max(0, 0.3 - (distance / 1000))
            
            pitch_std = features.get('pitch_std', 20)
            if profile['pitch_variance'] == 'low' and pitch_std < 20:
                score += 0.2
            elif profile['pitch_variance'] == 'medium' and 20 <= pitch_std <= 40:
                score += 0.2
            elif profile['pitch_variance'] == 'high' and pitch_std > 40:
                score += 0.2
            
            tempo = features.get('tempo', 120)
            tempo_range = profile['tempo_range']
            
            if tempo_range[0] <= tempo <= tempo_range[1]:
                score += 0.2
            else:
                distance = min(
                    abs(tempo - tempo_range[0]),
                    abs(tempo - tempo_range[1])
                )
                score += max(0, 0.2 - (distance / 50))
            
            mfcc_score = self._calculate_mfcc_similarity(features.get('mfcc_mean', np.zeros(13)), accent)
            score += mfcc_score * 0.3
            
            scores[accent] = max(0, min(1, score))
        
        return scores
    
    def _calculate_mfcc_similarity(self, mfcc_features, accent):
        accent_patterns = {
            'American': [0.2, -0.1, 0.3, -0.2, 0.1, -0.1, 0.2, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1],
            'British': [0.1, -0.2, 0.2, -0.3, 0.2, -0.1, 0.1, -0.2, 0.1, -0.1, 0.2, -0.1, 0.1],
            'Australian': [0.3, -0.1, 0.1, -0.2, 0.3, -0.1, 0.2, -0.1, 0.2, -0.1, 0.1, -0.2, 0.1],
            'Indian': [0.1, -0.3, 0.4, -0.1, 0.2, -0.2, 0.3, -0.1, 0.1, -0.2, 0.2, -0.1, 0.2],
            'Canadian': [0.2, -0.1, 0.2, -0.2, 0.1, -0.1, 0.1, -0.1, 0.2, -0.1, 0.1, -0.1, 0.1]
        }
        
        if accent not in accent_patterns:
            return 0
        
        try:
            pattern = np.array(accent_patterns[accent])
            mfcc_array = np.array(mfcc_features)
            
            mfcc_norm = np.linalg.norm(mfcc_array)
            pattern_norm = np.linalg.norm(pattern)
            
            if mfcc_norm > 0 and pattern_norm > 0:
                mfcc_normalized = mfcc_array / mfcc_norm
                pattern_normalized = pattern / pattern_norm
                
                similarity = np.dot(mfcc_normalized, pattern_normalized)
                return max(0, float(similarity))
            else:
                return 0.5
                
        except Exception as e:
            return 0.5
    
    def predict_accent(self, audio_path):
        if not os.path.exists(audio_path):
            return None
        
        features = self.extract_acoustic_features(audio_path)
        if not features:
            return None
        
        scores = self.calculate_accent_scores(features)
        
        total_score = sum(scores.values())
        if total_score > 0:
            normalized_scores = {k: v/total_score for k, v in scores.items()}
        else:
            normalized_scores = {k: 1.0/len(scores) for k in scores.keys()}
        
        predicted_accent = max(normalized_scores, key=normalized_scores.get)
        confidence = normalized_scores[predicted_accent]
        
        return {
            'accent': predicted_accent,
            'confidence': confidence,
            'all_probabilities': normalized_scores,
            'raw_scores': scores
        }
    
    def print_detailed_results(self, result):
        if not result:
            return
        
        print(f"Predicted Accent: {result['accent']}")
        print(f"Confidence Score: {result['confidence']:.1%}")
        
        print("All Accent Probabilities:")
        
        sorted_probs = sorted(
            result['all_probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for i, (accent, prob) in enumerate(sorted_probs):
            bar_length = int(prob * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"{accent:12}: {prob:.1%} |{bar}|")

def main():
    if len(sys.argv) != 2:
        print("Usage: python accent_classifier.py audio_file.mp3")
        return
    
    audio_file = sys.argv[1]
    
    classifier = SimpleOfflineAccentClassifier()
    result = classifier.predict_accent(audio_file)
    classifier.print_detailed_results(result)

if __name__ == "__main__":
    main()