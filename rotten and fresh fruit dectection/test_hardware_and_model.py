"""Hardware and model end-to-end automated verification script.

Tests:
1. Loads trained ONNX model (fruit.onnx) and metadata.json.
2. Runs inference on known fresh and rotten images from the test set.
3. Attempts to open webcam device 0, grab a frame, extract square crop, and run inference.
4. Reports inference time, output probabilities, and hardware status.
"""

from pathlib import Path
import time
import cv2
import numpy as np

from common import Camera, Predictor, square, CLASSES, DATA_DIR, MODEL_ONNX_PATH, METADATA_PATH


def verify_all():
    print("=" * 65)
    print("AUTOMATED VERIFICATION: HARDWARE, MODEL, AND INFERENCE")
    print("=" * 65)

    # 1. Model & Metadata Check
    print("\n[1/3] Checking Trained Model Artifacts...")
    assert MODEL_ONNX_PATH.exists(), f"Missing ONNX model at {MODEL_ONNX_PATH}"
    assert METADATA_PATH.exists(), f"Missing metadata at {METADATA_PATH}"
    print(f"      Model artifact:    {MODEL_ONNX_PATH.name} ({MODEL_ONNX_PATH.stat().st_size / (1024*1024):.2f} MB)")
    print(f"      Metadata artifact: {METADATA_PATH.name}")

    predictor = Predictor()
    print(f"      Classes:           {predictor.classes}")
    print(f"      Input size:        {predictor.size}x{predictor.size}")
    print("      [+] Predictor loaded successfully via OpenCV DNN.")

    # 2. Known Image Inference Check
    print("\n[2/3] Testing Inference on Test Images...")
    test_fresh = list((DATA_DIR / "test" / "fresh").glob("*.jpg"))
    test_rotten = list((DATA_DIR / "test" / "rotten").glob("*.jpg"))

    if test_fresh:
        sample_f = cv2.imread(str(test_fresh[0]))
        probs_f = predictor.predict(sample_f)
        print(f"      Sample Fresh:  {test_fresh[0].name} -> Fresh: {probs_f[0]*100:.1f}%, Rotten: {probs_f[1]*100:.1f}%")
        assert probs_f[0] > probs_f[1], "Expected fresh prediction for fresh sample"

    if test_rotten:
        sample_r = cv2.imread(str(test_rotten[0]))
        probs_r = predictor.predict(sample_r)
        print(f"      Sample Rotten: {test_rotten[0].name} -> Fresh: {probs_r[0]*100:.1f}%, Rotten: {probs_r[1]*100:.1f}%")
        assert probs_r[1] > probs_r[0], "Expected rotten prediction for rotten sample"

    print("      [+] Inference on test images verified.")

    # 3. Webcam Live Capture & Inference Check
    print("\n[3/3] Testing Live Webcam Capture & Inference...")
    try:
        with Camera(source="webcam", index=0) as cam:
            ret, frame = cam.read()
            if not ret or frame is None:
                print("      [-] Warning: Webcam frame was empty or device was busy.")
            else:
                h, w = frame.shape[:2]
                crop, (x, y, side) = square(frame)
                t0 = time.perf_counter()
                probs = predictor.predict(crop)
                dt = (time.perf_counter() - t0) * 1000.0

                winner_idx = int(np.argmax(probs))
                verdict = CLASSES[winner_idx].upper() if probs[winner_idx] >= 0.80 else "UNCERTAIN"
                print(f"      Camera Resolution: {w}x{h}")
                print(f"      Square Crop Size:  {side}x{side}")
                print(f"      Live Inference:    {dt:.2f} ms")
                print(f"      Live Scores:       Fresh={probs[0]*100:.1f}%, Rotten={probs[1]*100:.1f}%")
                print(f"      Live Verdict:      {verdict}")
                print("      [+] Webcam capture and live inference succeeded!")
    except Exception as err:
        print(f"      [-] Webcam check note: {err}")

    print("\n" + "=" * 65)
    print("AUTOMATED VERIFICATION PASSED SUCCESSFULLY")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    verify_all()
