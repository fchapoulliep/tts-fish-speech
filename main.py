import json
from pathlib import Path
from fishaudio import FishAudio
from fishaudio.utils import save

# CONFIGURATION
API_KEY = "72acd58fd2ed4620836532320e04d4ff"
VOICE_ID = "9a4b0964edb940e8b535f3de5129c388"
INPUT_FILE = "text_to_do.json"
OUTPUT_DIR = Path("output")


def main():
    # 1. Initialisation
    client = FishAudio(api_key=API_KEY)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 2. Chargement du JSON
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Erreur : Le fichier {INPUT_FILE} est introuvable.")
        return

    print(f"Début de la génération pour {len(data)} segments...")

    # 3. Boucle de génération
    for entry in data:
        file_id = entry.get("id", "audio")
        text = entry.get("text", "")
        file_path = OUTPUT_DIR / f"{file_id}.wav"

        if not text:
            continue

        print(f"Génération de '{file_id}'...")

        try:
            audio_bytes = client.tts.convert(
                text=text, reference_id=VOICE_ID, format="wav"
            )
            save(audio_bytes, str(file_path))
            print(f"  ✓ Sauvegardé dans {file_path}")
        except Exception as e:
            print(f"  ✗ Erreur sur '{file_id}': {e}")

    print("\nTraitement terminé !")


if __name__ == "__main__":
    main()
