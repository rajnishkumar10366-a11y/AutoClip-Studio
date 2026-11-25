# AutoClip-Studio
AI powered automatic viral video clip generator
# 🎬 AutoClip Studio

**AI-Powered Automatic Viral Video Clip Generator**

AutoClip Studio converts long-form videos into short-form clips for YouTube Shorts, Instagram Reels and TikTok.

## ✨ Features

- YouTube video URL support
- Upload your own video
- Automatic speech transcription using Faster-Whisper
- AI-powered clip selection using Ollama
- Automatic video cutting with FFmpeg
- Multiple aspect ratios: 9:16, 1:1, 16:9 and 4:5
- Multiple caption styles and font options
- Customizable caption positioning
- MP4 downloads and ZIP export
- Progress tracking during processing

## 🛠️ Technologies Used

- Python
- Streamlit
- Faster-Whisper
- Ollama
- FFmpeg
- yt-dlp

## 💻 Installation

### 1. Clone the repository

```bash
git clone https://github.com/rajnishkumar10366-a11y/AutoClip-Studio.git
```

### 2. Open the project folder

```bash
cd AutoClip-Studio
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install FFmpeg

Make sure FFmpeg and FFprobe are installed and available in PATH.

### 5. Run the application

```bash
python -m streamlit run autoclip.py
```

## 🤖 Optional AI Setup

Install Ollama and download a compatible model:

```bash
ollama pull llama3.2
```

Start Ollama before running AI-based clip ranking.

## 🚀 Project Goal

To make video repurposing easier by automatically finding interesting moments in long videos and converting them into short-form content.

## ⚠️ Requirements

- Python
- FFmpeg
- Internet connection for YouTube downloads
- Ollama for local AI ranking (optional)

## 👨‍💻 Developer

**Rajnish Kumar**

GitHub: [rajnishkumar10366-a11y](https://github.com/rajnishkumar10366-a11y)

---

⭐ If you find this project useful, consider giving it a star!
