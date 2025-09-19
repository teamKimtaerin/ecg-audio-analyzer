#!/usr/bin/env python3
"""
Fix WhisperX 3.1.1 TranscriptionOptions bug
This script patches the whisperx/asr.py file to add missing parameters
"""

import sys

def patch_whisperx():
    try:
        # Read the file
        filepath = '/usr/local/lib/python3.10/dist-packages/whisperx/asr.py'
        with open(filepath, 'r') as f:
            content = f.read()

        # Find and replace the problematic line
        old_line = 'default_asr_options = faster_whisper.transcribe.TranscriptionOptions(**default_asr_options)'

        # Check if already patched
        if 'repetition_penalty' in content:
            print("WhisperX already patched")
            return

        # Add missing parameters before the TranscriptionOptions call
        patch = """    # Patch: Add missing parameters for WhisperX 3.1.1
    default_asr_options['repetition_penalty'] = 1.0
    default_asr_options['no_repeat_ngram_size'] = 0
    default_asr_options['prompt_reset_on_temperature'] = 0.5
    default_asr_options = faster_whisper.transcribe.TranscriptionOptions(**default_asr_options)"""

        # Replace the line
        content = content.replace(old_line, patch)

        # Write back
        with open(filepath, 'w') as f:
            f.write(content)

        print("WhisperX patched successfully")

    except Exception as e:
        print(f"Error patching WhisperX: {e}")
        sys.exit(1)

if __name__ == "__main__":
    patch_whisperx()