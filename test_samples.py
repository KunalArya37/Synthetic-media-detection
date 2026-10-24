"""Quick validation script for sample images"""
from PIL import Image
from src.models.detector import build_detector
from src.engine.authenticator import MediaAuthenticator
from src.config import BEST_MODEL_PATH, SAMPLES_DIR, DEVICE

def main():
    print(f"Loading model on {DEVICE}...")
    model = build_detector(checkpoint_path=str(BEST_MODEL_PATH), device=DEVICE)
    auth = MediaAuthenticator(model)
    
    print("\n" + "="*80)
    print(f"{'Sample Filename':<35} | {'Verdict':<32} | {'Synth Prob':<10} | {'Conf'}")
    print("="*80)
    
    for p in sorted(list(SAMPLES_DIR.glob("*.jpg"))):
        img = Image.open(p)
        res = auth.authenticate(img)
        verdict = res["verdict"]
        synth_prob = f"{res['synthetic_probability']:.1f}%"
        conf = f"{res['confidence']:.1f}%"
        print(f"{p.name:<35} | {verdict:<32} | {synth_prob:<10} | {conf}")
    print("="*80)

if __name__ == "__main__":
    main()
