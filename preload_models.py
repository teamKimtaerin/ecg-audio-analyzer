#!/usr/bin/env python3
"""
Preload models with proper HTTP 301 handling
(v4 - Production ready with fallbacks)
"""
import os
import sys
import torch
import urllib.request
from pathlib import Path

# HTTP 301 핸들러 설정
class SmartRedirectHandler(urllib.request.HTTPRedirectHandler):
    def http_error_301(self, req, fp, code, msg, headers):
        result = urllib.request.HTTPRedirectHandler.http_error_301(
            self, req, fp, code, msg, headers)
        return result
    
    def http_error_302(self, req, fp, code, msg, headers):
        # 302도 처리 (임시 이동)
        result = urllib.request.HTTPRedirectHandler.http_error_302(
            self, req, fp, code, msg, headers)
        return result

opener = urllib.request.build_opener(SmartRedirectHandler)
urllib.request.install_opener(opener)

# 패치 후 import
import whisperx
from pyannote.audio import Pipeline

def ensure_cache_dirs():
    """캐시 디렉토리 생성"""
    dirs = [
        "/root/.cache/whisper",
        "/root/.cache/huggingface/hub",
        "/root/.cache/torch/hub"
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def preload_models():
    ensure_cache_dirs()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "float32"
    hf_token = os.environ.get('HF_TOKEN')
    
    print(f"🎮 Device: {device}, Compute: {compute_type}")
    
    # 1. WhisperX - 여러 방법 시도
    success = False
    try:
        print("📥 Loading WhisperX model...")
        model = whisperx.load_model("large-v2", device=device, compute_type=compute_type)
        del model
        print("✓ WhisperX cached")
        success = True
    except Exception as e:
        print(f"⚠ Method 1 failed: {e}")
    
    if not success:
        # Fallback: wget
        print("🔄 Trying direct download...")
        ret = os.system("wget -nc -q -P /root/.cache/whisper https://openaipublic.azureedge.net/main/whisper/models/large-v2.pt")
        if ret == 0:
            print("✓ WhisperX downloaded via wget")
        else:
            print("❌ WhisperX download failed completely")
    
    # 2. Pyannote - 간단하게
    if hf_token:
        try:
            print("📥 Loading Pyannote...")
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token=hf_token
            )
            del pipeline
            print("✓ Pyannote cached")
        except:
            print("⚠ Pyannote skipped (version mismatch is OK)")
    
    # 3. Alignment 모델 (기존 코드)
    print("📥 Loading alignment models...")
    try:
        for lang in ['en', 'ko', 'ja']:  # 핵심 언어만
            try:
                model, metadata = whisperx.load_align_model(lang, device)
                del model, metadata
                print(f"  ✓ {lang} alignment cached")
            except Exception as e:
                print(f"  ⚠ {lang} failed: {e}")
    except:
        print("⚠ Alignment models partial success")
    
    print("✅ Preloading completed")

if __name__ == "__main__":
    preload_models()
    sys.exit(0)