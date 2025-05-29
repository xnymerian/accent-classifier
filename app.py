from flask import Flask, render_template, request, jsonify
import yt_dlp
import os
import tempfile
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
        
        # Geçici dosya oluştur
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_audio_path = temp_file.name
        
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
            },
            'cookiesfrombrowser': ('chrome',),  # Tarayıcı çerezlerini kullan
            'extract_flat': 'in_playlist',  # Playlist içindeyse sadece ilk videoyu al
            'noplaylist': True,  # Playlist değil
            'ignoreerrors': True,  # Hataları görmezden gel
            'no_check_certificate': True,  # Sertifika kontrolünü atla
            'prefer_insecure': True,  # Güvenli olmayan bağlantıları tercih et
            'http_headers': {  # Tarayıcı gibi görün
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
        # Hata detayını logla
        print(f"Error: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(debug=True, host='0.0.0.0', port=port)
