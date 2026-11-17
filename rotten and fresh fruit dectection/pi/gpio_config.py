"""Raspberry Pi GPIO Status LED Controller.

Controls 3 optional status indicator LEDs:
- Green LED:  FRESH result
- Yellow LED: UNCERTAIN result
- Red LED:    ROTTEN result

All pins are cleanly reset and turned OFF on program exit, batch reset, or camera failure.
"""

from __future__ import annotations

import atexit
import logging
import sys
from typing import Optional

logger = logging.getLogger("FruitFreshness.GPIO")

# Default BCM GPIO Pin Numbers
DEFAULT_PIN_GREEN = 17   # Pin 11 on header
DEFAULT_PIN_YELLOW = 27  # Pin 13 on header
DEFAULT_PIN_RED = 22     # Pin 15 on header


class LedController:
    """Controls physical status LEDs via Raspberry Pi GPIO.

    Gracefully falls back if running on non-Pi systems or if GPIO libraries are missing.
    """

    def __init__(
        self,
        pin_green: int = DEFAULT_PIN_GREEN,
        pin_yellow: int = DEFAULT_PIN_YELLOW,
        pin_red: int = DEFAULT_PIN_RED,
    ):
        self.pin_green = pin_green
        self.pin_yellow = pin_yellow
        self.pin_red = pin_red
        self.initialized = False
        self._led_green = None
        self._led_yellow = None
        self._led_red = None
        self._backend = None

        self._setup()
        atexit.register(self.cleanup)

    def _setup(self) -> None:
        """Initialize GPIO pins using gpiozero or RPi.GPIO."""
        # Try gpiozero first (standard on Raspberry Pi OS Bookworm)
        try:
            from gpiozero import LED  # type: ignore

            self._led_green = LED(self.pin_green)
            self._led_yellow = LED(self.pin_yellow)
            self._led_red = LED(self.pin_red)
            self._backend = "gpiozero"
            self.initialized = True
            self.all_off()
            logger.info("GPIO LEDs initialized via gpiozero (G:%d, Y:%d, R:%d)", self.pin_green, self.pin_yellow, self.pin_red)
            return
        except (ImportError, Exception) as e:
            logger.debug("gpiozero not available: %s. Trying RPi.GPIO...", e)

        # Fallback to RPi.GPIO
        try:
            import RPi.GPIO as GPIO  # type: ignore

            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(self.pin_green, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(self.pin_yellow, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(self.pin_red, GPIO.OUT, initial=GPIO.LOW)
            self._backend = "RPi.GPIO"
            self.initialized = True
            self.all_off()
            logger.info("GPIO LEDs initialized via RPi.GPIO (G:%d, Y:%d, R:%d)", self.pin_green, self.pin_yellow, self.pin_red)
            return
        except (ImportError, Exception) as e:
            logger.debug("RPi.GPIO not available: %s", e)

        raise RuntimeError(
            "Neither 'gpiozero' nor 'RPi.GPIO' is available on this system.\n"
            "Problem: Cannot control physical LEDs.\n"
            "Cause: This script is either running on a non-Pi machine or GPIO libraries are uninstalled.\n"
            "Fix: On Raspberry Pi, install with 'sudo apt install python3-gpiozero python3-rpi.gpio'."
        )

    def set_result(self, verdict: str) -> None:
        """Light up the corresponding LED based on the inspection verdict.

        Args:
            verdict: 'FRESH', 'ROTTEN', or 'UNCERTAIN'.
        """
        if not self.initialized:
            return

        v = verdict.upper()
        g_val = (v == "FRESH")
        y_val = (v == "UNCERTAIN")
        r_val = (v == "ROTTEN")

        if self._backend == "gpiozero":
            if g_val:
                self._led_green.on()
            else:
                self._led_green.off()

            if y_val:
                self._led_yellow.on()
            else:
                self._led_yellow.off()

            if r_val:
                self._led_red.on()
            else:
                self._led_red.off()

        elif self._backend == "RPi.GPIO":
            import RPi.GPIO as GPIO  # type: ignore

            GPIO.output(self.pin_green, GPIO.HIGH if g_val else GPIO.LOW)
            GPIO.output(self.pin_yellow, GPIO.HIGH if y_val else GPIO.LOW)
            GPIO.output(self.pin_red, GPIO.HIGH if r_val else GPIO.LOW)

    def all_off(self) -> None:
        """Turn off all 3 indicator LEDs."""
        if not self.initialized:
            return

        if self._backend == "gpiozero":
            if self._led_green:
                self._led_green.off()
            if self._led_yellow:
                self._led_yellow.off()
            if self._led_red:
                self._led_red.off()
        elif self._backend == "RPi.GPIO":
            import RPi.GPIO as GPIO  # type: ignore

            GPIO.output(self.pin_green, GPIO.LOW)
            GPIO.output(self.pin_yellow, GPIO.LOW)
            GPIO.output(self.pin_red, GPIO.LOW)

    def cleanup(self) -> None:
        """Safely extinguish LEDs and release GPIO pins."""
        if not self.initialized:
            return

        try:
            self.all_off()
            if self._backend == "gpiozero":
                if self._led_green:
                    self._led_green.close()
                if self._led_yellow:
                    self._led_yellow.close()
                if self._led_red:
                    self._led_red.close()
            elif self._backend == "RPi.GPIO":
                import RPi.GPIO as GPIO  # type: ignore

                GPIO.cleanup([self.pin_green, self.pin_yellow, self.pin_red])
        except Exception:
            pass
        finally:
            self.initialized = False
