from flask import Flask, render_template, request, jsonify
import yt_dlp
import os
from detect import SimpleOfflineAccentClassifier

app = Flask(__name__)
classifier = SimpleOfflineAccentClassifier()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        video_url = request.form['url']
        
        # YouTube'dan ses indir
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'outtmpl': 'temp_audio',
            'quiet': True,
            'no_warnings': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        
        # Ses dosyasını analiz et
        result = classifier.predict_accent('temp_audio.wav')
        
        # Geçici dosyayı temizle
        if os.path.exists('temp_audio.wav'):
            os.remove('temp_audio.wav')
        
        if result is None:
            return jsonify({'error': 'voice analyze failed.'})
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)