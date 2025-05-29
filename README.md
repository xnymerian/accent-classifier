# AI Accent Classifier

A tool for analyzing English accents from audio files or video URLs. Perfect for evaluating spoken English in hiring processes.

##  Quick Start Guide

### 1. Setup

First, install the required packages:
```bash
pip install -r requirements.txt
```

Make sure you have FFmpeg installed on your system:
- Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html)
- Mac: `brew install ffmpeg`
- Linux: `sudo apt-get install ffmpeg`

### 2. Running the Application

You have three ways to use the tool:

#### A. Web Interface (Recommended)
```bash
python app.py
```
Then open: http://localhost:5000

ERROR!
Some functions are not working due to temporary permission issues in the YouTube bot or deployment environment. It will work without any problems locally.

Features:
- Upload audio files
- Paste YouTube URLs
- View detailed results

#### B. Gradio Interface (Alternative)
```bash
python detect2.py
```
Then open: http://localhost:7860

Features:
- Modern UI
- Real-time analysis
- Example files included

#### C. Command Line
```bash
python detect.py path/to/your/audio.mp3
```

### 3. Example Usage

 **Using YouTube URL:**
   - Go to http://localhost:5000
   - Paste a YouTube URL (e.g., https://www.youtube.com/watch?v=...)
   - Click "Analyze"
   - View results


 **Using Command Line:**
   ```bash
   python detect.py interview_recording.mp3
   ```

### Understanding Results

The tool provides:
- Predicted accent (e.g., British, American)
- Confidence score (0-100%)
- Probability distribution for all accents
- Visual representation of results

Example output:
