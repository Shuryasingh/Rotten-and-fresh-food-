"""Hardware diagnostic tool to test webcam or Raspberry Pi camera connectivity.

Runs completely independently of the ML model to ensure camera hardware and
drivers are operational before data collection or live scanning.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from common import Camera, square


def run_camera_test(source: str = "webcam", index: int = 0, width: int = 640, height: int = 480) -> None:
    """Run interactive camera preview window with diagnostic overlay."""
    print("=" * 60)
    print("FRUIT FRESHNESS DETECTOR - CAMERA HARDWARE DIAGNOSTIC")
    print("=" * 60)
    print(f"Connecting to camera source: '{source}', index: {index}, target res: {width}x{height}")

    try:
        camera = Camera(source=source, index=index, width=width, height=height)
    except Exception as err:
        print(f"\n[-] ERROR INITIALIZING CAMERA:\n{err}")
        sys.exit(1)

    window_name = "Fruit Freshness - Camera Hardware Test (Press Q to Quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n[+] Camera connected successfully!")
    print("    - A live preview window has opened.")
    print("    - The green square shows the inspection zone used for fruit classification.")
    print("    - Press 'Q' or 'ESC' in the preview window to exit.\n")

    frame_count = 0
    start_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                print("[-] Failed to grab camera frame.")
                break

            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            h, w = frame.shape[:2]
            cropped, (sq_x, sq_y, side) = square(frame)

            # Draw UI overlay
            display = frame.copy()

            # Semi-transparent dark banner at top
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)

            # Title & Status text
            cv2.putText(
                display,
                "CAMERA HARDWARE TEST",
                (15, 24),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6,
                (0, 255, 200),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                f"Res: {w}x{h}  |  FPS: {fps:.1f}  |  Source: {source}[{index}]",
                (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            # Draw inspection square in vibrant green
            cv2.rectangle(display, (sq_x, sq_y), (sq_x + side, sq_y + side), (0, 255, 0), 2)
            cv2.putText(
                display,
                "Inspection Zone (80% crop)",
                (sq_x + 5, sq_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            # Bottom instructions banner
            cv2.rectangle(display, (0, h - 30), (w, h), (20, 20, 20), -1)
            cv2.putText(
                display,
                "Press 'Q' or 'ESC' to close test",
                (15, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
                print("[*] Exit requested by operator.")
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("[+] Camera released cleanly. Test completed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test camera feed and verify hardware connectivity and crop geometry."
    )
    parser.add_argument(
        "--source",
        choices=["webcam", "pi"],
        default="webcam",
        help="Camera input source (default: webcam)",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Preferred capture width (default: 640)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Preferred capture height (default: 480)",
    )
    args = parser.parse_args()

    run_camera_test(source=args.source, index=args.index, width=args.width, height=args.height)


if __name__ == "__main__":
    main()
