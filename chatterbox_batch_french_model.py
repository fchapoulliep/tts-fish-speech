"""
Chatterbox TTS - Batch generation (French finetune)
Usage : python chatterbox_batch.py
"""

import gc
import json
from pathlib import Path
import torch
import torchaudio
from chatterbox.tts import ChatterboxTTS
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


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

    model = ChatterboxTTS.from_pretrained(device=device)
    ckpt_path = hf_hub_download(
        repo_id="Thomcles/Chatterbox-TTS-French", filename="t3_cfg.safetensors"
    )
    t3_state = load_file(ckpt_path, device="cpu")
    model.t3.load_state_dict(t3_state)
    model.t3.to(device).eval()

    total = len(tracks)
    for i, track in enumerate(tracks, 1):
        out_path = output_dir / f"{track['id']}.wav"
        print(f"[{i:02d}/{total}] {track['id']}")
        if out_path.exists():
            print("  ⏭️  déjà existant, ignoré.")
            continue
        with torch.inference_mode():
            wav = model.generate(track["text"], audio_prompt_path=sample_path)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        torchaudio.save(str(out_path), wav.cpu(), model.sr)
        print(f"  ✅ {out_path.name}")
        del wav
        gc.collect()

    print(f"\n✅ Terminé - fichiers dans {output_dir.resolve()}")


if __name__ == "__main__":
    main()
