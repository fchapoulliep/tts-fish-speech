"""
Chatterbox TTS - Batch generation (modèle multilingue)
Usage : uv run chatterbox_batch.py ma_voix.wav
"""

import json
from pathlib import Path
import torch
import torchaudio


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    sample_path = "ma_voix.wav"
    tracks_path = Path("tracks.json")
    output_dir = Path("outputs_tts")
    output_dir.mkdir(exist_ok=True)

    tracks = json.loads(tracks_path.read_text(encoding="utf-8"))
    print(f"📋 {len(tracks)} tracks chargés depuis {tracks_path}")

    device = get_device()
    print(f"🔧 Device : {device.upper()}")

    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    total = len(tracks)
    for i, track in enumerate(tracks, 1):
        out_path = output_dir / f"{track['id']}.wav"
        print(f"[{i:02d}/{total}] {track['id']}")
        if out_path.exists():
            print("  ⏭️  déjà existant, ignoré.")
            continue
        with torch.inference_mode():
            wav = model.generate(track["text"], audio_prompt_path=sample_path, language_id="fr")
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        torchaudio.save(str(out_path), wav.cpu(), model.sr)
        print(f"  ✅ {out_path.name}")

    print(f"\n✅ Terminé - fichiers dans {output_dir.resolve()}")


if __name__ == "__main__":
    main()