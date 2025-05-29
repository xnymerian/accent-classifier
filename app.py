from flask import Flask, render_template, request, jsonify
import yt_dlp
import os
from detect import SimpleOfflineAccentClassifier

app = Flask(__name__)
classifier = SimpleOfflineAccentClassifier()

# Geçici dosya dizini
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
os.makedirs(TEMP_DIR, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        video_url = request.form['url']
        
        # Geçici dosya yolu
        temp_audio_path = os.path.join(TEMP_DIR, 'temp_audio.wav')
        
        # YouTube'dan ses indir
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'outtmpl': temp_audio_path.replace('.wav', ''),
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'noplaylist': True,
            'ignoreerrors': True,
            'no_check_certificate': True,
            'prefer_insecure': True,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-us,en;q=0.5',
                'Sec-Fetch-Mode': 'navigate',
            }
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        
        # Ses dosyasının varlığını kontrol et
        if not os.path.exists(temp_audio_path):
            return jsonify({'error': 'Audio file could not be downloaded. Please try a different video.'})
        
        # Ses dosyasını analiz et
        result = classifier.predict_accent(temp_audio_path)
        
        # Geçici dosyayı temizle
        try:
            os.remove(temp_audio_path)
        except:
            pass
        
        if result is None:
            return jsonify({'error': 'Voice analysis failed. Please try a different video.'})
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port)
