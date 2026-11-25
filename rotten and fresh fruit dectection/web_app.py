"""Fruit Freshness Detector - Live Camera Web App.

Directly streams live camera video inside the browser with real-time AI classification:
- Live camera viewfinder directly in the web page via HTML5 getUserMedia or OpenCV stream.
- Center inspection target square with dynamic color feedback (Green=FRESH, Red=ROTTEN, Amber=UNCERTAIN).
- Continuous real-time MobileNetV2 ONNX inference on held-up fruit (Apples, Bananas, Oranges, Mangoes, Grapes, Strawberries, Avocados).
- Live probability meters, confidence score, and inference latency.
- Camera selector dropdown to switch between webcams.
- Log Item / Freeze button to commit scans to history table and export CSV.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import sys
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from common import (
    CLASSES,
    DATA_DIR,
    DEFAULT_THRESHOLD,
    MODEL_ONNX_PATH,
    Predictor,
    log_scan_event,
    square,
)

PORT = 8080

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Fruit Freshness AI • Live Camera Scanner</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-base: #07090e;
      --bg-surface: #0f141f;
      --bg-card: rgba(17, 24, 39, 0.78);
      --border-subtle: rgba(255, 255, 255, 0.08);
      --border-focus: rgba(6, 182, 212, 0.45);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --accent-fresh: #10b981;
      --accent-fresh-glow: rgba(16, 185, 129, 0.35);
      --accent-rotten: #ef4444;
      --accent-rotten-glow: rgba(239, 68, 68, 0.35);
      --accent-uncertain: #f59e0b;
      --accent-uncertain-glow: rgba(245, 158, 11, 0.35);
      --accent-cyan: #06b6d4;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: 'Inter', sans-serif;
      background: radial-gradient(circle at 50% 0%, #1e293b 0%, #07090e 65%);
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
    }

    /* Header */
    header {
      background: rgba(15, 20, 31, 0.85);
      backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border-subtle);
      position: sticky;
      top: 0;
      z-index: 50;
      padding: 14px 28px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .brand-icon {
      width: 40px;
      height: 40px;
      border-radius: 12px;
      background: linear-gradient(135deg, #10b981 0%, #06b6d4 100%);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 22px;
      box-shadow: 0 4px 16px rgba(16, 185, 129, 0.3);
    }

    .brand-title {
      font-family: 'Outfit', sans-serif;
      font-size: 19px;
      font-weight: 700;
      letter-spacing: -0.01em;
      background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .brand-sub {
      font-size: 12px;
      color: var(--text-muted);
    }

    .header-controls {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .badge-live {
      display: flex;
      align-items: center;
      gap: 7px;
      padding: 6px 14px;
      background: rgba(16, 185, 129, 0.12);
      border: 1px solid rgba(16, 185, 129, 0.35);
      border-radius: 20px;
      font-family: 'Outfit', sans-serif;
      font-size: 12px;
      font-weight: 700;
      color: #34d399;
      letter-spacing: 0.04em;
    }

    .live-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #10b981;
      box-shadow: 0 0 10px #10b981;
      animation: pulseDot 1.4s infinite ease-in-out;
    }

    @keyframes pulseDot {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.85); }
    }

    /* Main Grid */
    main {
      flex: 1;
      max-width: 1360px;
      width: 100%;
      margin: 0 auto;
      padding: 24px 20px;
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 24px;
    }

    @media (max-width: 1024px) {
      main {
        grid-template-columns: 1fr;
      }
    }

    .card {
      background: var(--bg-card);
      backdrop-filter: blur(24px);
      border: 1px solid var(--border-subtle);
      border-radius: 20px;
      padding: 20px;
      box-shadow: 0 12px 36px rgba(0, 0, 0, 0.45);
      display: flex;
      flex-direction: column;
      position: relative;
    }

    .card-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 14px;
    }

    .card-title {
      font-family: 'Outfit', sans-serif;
      font-size: 17px;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    /* Camera Viewfinder */
    .viewfinder-wrapper {
      position: relative;
      width: 100%;
      aspect-ratio: 4 / 3;
      max-height: 480px;
      background: #000;
      border-radius: 16px;
      overflow: hidden;
      border: 1px solid var(--border-subtle);
      box-shadow: inset 0 0 40px rgba(0, 0, 0, 0.8);
      display: flex;
      align-items: center;
      justify-content: center;
    }

    #videoFeed {
      width: 100%;
      height: 100%;
      object-fit: cover;
      transform: scaleX(-1); /* Mirror view for intuitive user experience */
    }

    /* Overlay Canvas */
    #overlayCanvas {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
    }

    /* Camera Controls Toolbelt */
    .camera-toolbar {
      margin-top: 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 10px;
    }

    .tool-group {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    select, button {
      font-family: 'Outfit', sans-serif;
      font-size: 13px;
      font-weight: 600;
      padding: 8px 14px;
      border-radius: 10px;
      border: 1px solid var(--border-subtle);
      background: rgba(255, 255, 255, 0.05);
      color: var(--text-main);
      cursor: pointer;
      outline: none;
      transition: all 0.2s ease;
    }

    select:hover, button:hover {
      background: rgba(255, 255, 255, 0.1);
      border-color: rgba(255, 255, 255, 0.2);
    }

    .btn-action {
      background: linear-gradient(135deg, #10b981 0%, #059669 100%);
      color: #fff;
      border: none;
      box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35);
    }

    .btn-action:hover {
      filter: brightness(1.1);
      transform: translateY(-1px);
    }

    /* Right Column: Live Classification Dashboard */
    .verdict-box {
      border-radius: 16px;
      padding: 24px 20px;
      text-align: center;
      transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
      position: relative;
      overflow: hidden;
    }

    .verdict-box.fresh {
      background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(6, 182, 212, 0.08) 100%);
      border: 2px solid #10b981;
      box-shadow: 0 0 35px var(--accent-fresh-glow);
    }

    .verdict-box.rotten {
      background: linear-gradient(135deg, rgba(239, 68, 68, 0.22) 0%, rgba(185, 28, 28, 0.08) 100%);
      border: 2px solid #ef4444;
      box-shadow: 0 0 35px var(--accent-rotten-glow);
    }

    .verdict-box.uncertain {
      background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(217, 119, 6, 0.08) 100%);
      border: 2px solid #f59e0b;
      box-shadow: 0 0 35px var(--accent-uncertain-glow);
    }

    .verdict-box.standby {
      background: rgba(255, 255, 255, 0.03);
      border: 1px dashed rgba(255, 255, 255, 0.15);
    }

    .verdict-badge-label {
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-muted);
      margin-bottom: 6px;
    }

    .verdict-state {
      font-family: 'Outfit', sans-serif;
      font-size: 44px;
      font-weight: 800;
      letter-spacing: -0.01em;
      line-height: 1.1;
      margin-bottom: 6px;
    }

    .verdict-box.fresh .verdict-state { color: #34d399; }
    .verdict-box.rotten .verdict-state { color: #f87171; }
    .verdict-box.uncertain .verdict-state { color: #fbbf24; }
    .verdict-box.standby .verdict-state { color: var(--text-dim); font-size: 28px; }

    .verdict-sub {
      font-size: 14px;
      color: #e2e8f0;
    }

    /* Meters */
    .metrics-card {
      margin-top: 18px;
      background: rgba(15, 23, 42, 0.5);
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .meter-block {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .meter-label {
      display: flex;
      justify-content: space-between;
      font-size: 13px;
      font-weight: 600;
    }

    .meter-track {
      width: 100%;
      height: 12px;
      background: rgba(255, 255, 255, 0.06);
      border-radius: 6px;
      overflow: hidden;
      position: relative;
    }

    .meter-fill {
      height: 100%;
      border-radius: 6px;
      transition: width 0.25s ease-out;
    }

    .meter-fill.fresh {
      background: linear-gradient(90deg, #059669, #10b981);
      box-shadow: 0 0 12px rgba(16, 185, 129, 0.5);
    }

    .meter-fill.rotten {
      background: linear-gradient(90deg, #dc2626, #ef4444);
      box-shadow: 0 0 12px rgba(239, 68, 68, 0.5);
    }

    /* Diagnostics pill bar */
    .stats-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 14px;
    }

    .stat-pill {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--border-subtle);
      border-radius: 10px;
      padding: 10px 12px;
      display: flex;
      flex-direction: column;
      gap: 2px;
    }

    .stat-title {
      font-size: 11px;
      color: var(--text-dim);
      text-transform: uppercase;
      font-weight: 600;
      letter-spacing: 0.04em;
    }

    .stat-value {
      font-family: 'Outfit', sans-serif;
      font-size: 16px;
      font-weight: 700;
      color: var(--text-main);
    }

    /* Threshold slider */
    .slider-card {
      margin-top: 14px;
      padding: 14px 16px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .slider-header {
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      color: var(--text-muted);
      font-weight: 500;
    }

    input[type=range] {
      width: 100%;
      height: 6px;
      border-radius: 3px;
      background: #1e293b;
      outline: none;
      -webkit-appearance: none;
      accent-color: var(--accent-cyan);
    }

    /* Logged History Table */
    .history-card {
      grid-column: 1 / -1;
      margin-top: 4px;
    }

    .history-table-wrapper {
      max-height: 220px;
      overflow-y: auto;
      margin-top: 10px;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }

    th {
      text-align: left;
      padding: 8px 12px;
      background: rgba(15, 23, 42, 0.7);
      color: var(--text-muted);
      font-weight: 600;
      position: sticky;
      top: 0;
      z-index: 5;
      border-bottom: 1px solid var(--border-subtle);
    }

    td {
      padding: 8px 12px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }

    .badge-status {
      display: inline-block;
      font-family: 'Outfit', sans-serif;
      font-size: 11px;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 6px;
      text-transform: uppercase;
    }

    .badge-status.fresh { background: rgba(16, 185, 129, 0.15); color: #34d399; }
    .badge-status.rotten { background: rgba(239, 68, 68, 0.15); color: #f87171; }
    .badge-status.uncertain { background: rgba(245, 158, 11, 0.15); color: #fbbf24; }

    footer {
      text-align: center;
      padding: 20px;
      font-size: 12px;
      color: var(--text-dim);
      border-top: 1px solid var(--border-subtle);
      background: #05070a;
    }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="brand-icon">🍎</div>
      <div>
        <div class="brand-title">Fruit Freshness AI Detector</div>
        <div class="brand-sub">Real-Time Camera Scanner • MobileNetV2 ONNX</div>
      </div>
    </div>
    <div class="header-controls">
      <div class="badge-live">
        <div class="live-dot"></div>
        <span>LIVE INFERENCE ACTIVE</span>
      </div>
    </div>
  </header>

  <main>
    <!-- Left: Live Video Viewfinder -->
    <div class="card">
      <div class="card-header">
        <div class="card-title">
          <span>📹 Live Camera Viewfinder</span>
        </div>
        <span id="fpsBadge" style="font-size: 12px; color: var(--text-muted);">Inference: ~2.5 FPS</span>
      </div>

      <div class="viewfinder-wrapper">
        <video id="videoFeed" playsinline autoplay muted></video>
        <canvas id="overlayCanvas"></canvas>
      </div>

      <div class="camera-toolbar">
        <div class="tool-group">
          <label style="font-size: 12px; color: var(--text-muted);">Camera:</label>
          <select id="cameraSelect" onchange="switchCamera(this.value)">
            <option value="">Default Webcam</option>
          </select>
          <button onclick="toggleStream()" id="btnStreamToggle">⏸ Pause Camera</button>
        </div>

        <div class="tool-group">
          <button class="btn-action" onclick="logCurrentItem()" id="btnLog">
            <span>💾 Log Current Item (Space)</span>
          </button>
        </div>
      </div>
    </div>

    <!-- Right: Real-Time Verdict & Probabilities -->
    <div class="card">
      <div class="card-header">
        <div class="card-title">
          <span>🎯 Real-Time Inspection Diagnosis</span>
        </div>
        <span id="scanIntervalBadge" style="font-size: 12px; color: var(--text-cyan);">Interval: 400ms</span>
      </div>

      <div class="verdict-box standby" id="verdictBox">
        <div class="verdict-badge-label">AI Status</div>
        <div class="verdict-state" id="verdictText">INITIALIZING...</div>
        <div class="verdict-sub" id="verdictSub">Hold an apple, banana, orange, mango, or grapes in front of the camera</div>
      </div>

      <div class="metrics-card">
        <div class="meter-block">
          <div class="meter-label">
            <span style="color: #34d399;">Fresh Score</span>
            <span id="freshPct">0.0%</span>
          </div>
          <div class="meter-track">
            <div class="meter-fill fresh" id="freshBar" style="width: 0%;"></div>
          </div>
        </div>

        <div class="meter-block">
          <div class="meter-label">
            <span style="color: #f87171;">Rotten Score</span>
            <span id="rottenPct">0.0%</span>
          </div>
          <div class="meter-track">
            <div class="meter-fill rotten" id="rottenBar" style="width: 0%;"></div>
          </div>
        </div>
      </div>

      <div class="stats-row">
        <div class="stat-pill">
          <span class="stat-title">Inference Time</span>
          <span class="stat-value" id="latencyStat">-- ms</span>
        </div>
        <div class="stat-pill">
          <span class="stat-title">Items Logged</span>
          <span class="stat-value" id="loggedCountStat">0</span>
        </div>
      </div>

      <div class="slider-card">
        <div class="slider-header">
          <span>Certainty Threshold</span>
          <span id="thresholdVal" style="font-weight: 700; color: #fff;">80%</span>
        </div>
        <input type="range" id="thresholdSlider" min="50" max="95" value="80" oninput="updateThreshold(this.value)">
      </div>
    </div>

    <!-- Full-Width: Inspection History Table -->
    <div class="card history-card">
      <div class="card-header">
        <div class="card-title">
          <span>📋 Session Item Log</span>
        </div>
        <button onclick="downloadCSV()">📥 Download CSV Log</button>
      </div>

      <div class="history-table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Verdict</th>
              <th>Confidence</th>
              <th>Fresh Score</th>
              <th>Rotten Score</th>
              <th>Latency</th>
            </tr>
          </thead>
          <tbody id="historyTbody">
            <tr>
              <td colspan="6" style="text-align: center; color: var(--text-dim); padding: 20px;">
                Hold fruit inside the square and press "Log Current Item" or Spacebar.
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </main>

  <footer>
    Fruit Freshness AI Detector • MobileNetV2 ONNX • Trained on Apples, Bananas, Oranges, Mangoes & Grapes
  </footer>

  <!-- Hidden Processing Canvas for Crop Frame Grabs -->
  <canvas id="cropCanvas" width="224" height="224" style="display: none;"></canvas>

  <script>
    let currentStream = null;
    let isStreamActive = true;
    let isProcessing = false;
    let certaintyThreshold = 0.80;
    let scanIntervalMs = 400;
    let lastResult = null;
    let historyRecords = [];
    let intervalTimer = null;

    const video = document.getElementById('videoFeed');
    const overlay = document.getElementById('overlayCanvas');
    const overlayCtx = overlay.getContext('2d');
    const cropCanvas = document.getElementById('cropCanvas');
    const cropCtx = cropCanvas.getContext('2d');

    // 1. Initialize Camera Stream
    async function startCamera(deviceId = null) {
      if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
      }

      const constraints = {
        video: deviceId ? { deviceId: { exact: deviceId } } : { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }
      };

      try {
        currentStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = currentStream;
        await video.play();

        // Enumerate devices for camera dropdown
        enumerateCameras();

        // Start drawing loop and analysis interval
        requestAnimationFrame(drawOverlayLoop);
        if (!intervalTimer) {
          intervalTimer = setInterval(captureAndClassify, scanIntervalMs);
        }
      } catch (err) {
        console.error("Webcam access error:", err);
        document.getElementById('verdictText').innerText = "CAMERA BLOCKED";
        document.getElementById('verdictSub').innerText = "Please allow camera access in browser permissions or check if another app is using it.";
      }
    }

    async function enumerateCameras() {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(d => d.kind === 'videoinput');
        const select = document.getElementById('cameraSelect');
        const currentVal = select.value;
        select.innerHTML = '';

        videoDevices.forEach((dev, idx) => {
          const opt = document.createElement('option');
          opt.value = dev.deviceId;
          opt.text = dev.label || `Camera ${idx + 1}`;
          select.appendChild(opt);
        });

        if (currentVal && select.querySelector(`option[value="${currentVal}"]`)) {
          select.value = currentVal;
        }
      } catch (e) {
        console.warn("Could not list video devices:", e);
      }
    }

    function switchCamera(deviceId) {
      if (deviceId) {
        startCamera(deviceId);
      }
    }

    function toggleStream() {
      const btn = document.getElementById('btnStreamToggle');
      if (isStreamActive) {
        if (currentStream) {
          currentStream.getVideoTracks().forEach(t => t.enabled = false);
        }
        isStreamActive = false;
        btn.innerText = "▶ Resume Camera";
      } else {
        if (currentStream) {
          currentStream.getVideoTracks().forEach(t => t.enabled = true);
        }
        isStreamActive = true;
        btn.innerText = "⏸ Pause Camera";
      }
    }

    // 2. Draw Inspection Square Overlay on Top of Video Feed
    function drawOverlayLoop() {
      if (video.videoWidth > 0) {
        if (overlay.width !== video.clientWidth || overlay.height !== video.clientHeight) {
          overlay.width = video.clientWidth;
          overlay.height = video.clientHeight;
        }

        const w = overlay.width;
        const h = overlay.height;
        overlayCtx.clearRect(0, 0, w, h);

        // Center square calculation (80% of minor dimension)
        const side = Math.floor(Math.min(w, h) * 0.75);
        const x = Math.floor((w - side) / 2);
        const y = Math.floor((h - side) / 2);

        // Dynamic box color based on last verdict
        let boxColor = 'rgba(6, 182, 212, 0.7)'; // Cyan standby
        if (lastResult) {
          if (lastResult.verdict === 'FRESH') boxColor = '#10b981';
          else if (lastResult.verdict === 'ROTTEN') boxColor = '#ef4444';
          else if (lastResult.verdict === 'UNCERTAIN') boxColor = '#f59e0b';
        }

        // Draw darkened backdrop outside inspection square
        overlayCtx.fillStyle = 'rgba(0, 0, 0, 0.35)';
        overlayCtx.fillRect(0, 0, w, y); // Top
        overlayCtx.fillRect(0, y + side, w, h - (y + side)); // Bottom
        overlayCtx.fillRect(0, y, x, side); // Left
        overlayCtx.fillRect(x + side, y, w - (x + side), side); // Right

        // Draw Inspection Square Bounding Box
        overlayCtx.strokeStyle = boxColor;
        overlayCtx.lineWidth = 2.5;
        overlayCtx.strokeRect(x, y, side, side);

        // Draw stylish corner brackets
        const cornerLen = 20;
        overlayCtx.lineWidth = 4;
        overlayCtx.strokeStyle = boxColor;

        // Top-left
        overlayCtx.beginPath();
        overlayCtx.moveTo(x, y + cornerLen);
        overlayCtx.lineTo(x, y);
        overlayCtx.lineTo(x + cornerLen, y);
        overlayCtx.stroke();

        // Top-right
        overlayCtx.beginPath();
        overlayCtx.moveTo(x + side - cornerLen, y);
        overlayCtx.lineTo(x + side, y);
        overlayCtx.lineTo(x + side, y + cornerLen);
        overlayCtx.stroke();

        // Bottom-left
        overlayCtx.beginPath();
        overlayCtx.moveTo(x, y + side - cornerLen);
        overlayCtx.lineTo(x, y + side);
        overlayCtx.lineTo(x + cornerLen, y + side);
        overlayCtx.stroke();

        // Bottom-right
        overlayCtx.beginPath();
        overlayCtx.moveTo(x + side - cornerLen, y + side);
        overlayCtx.lineTo(x + side, y + side);
        overlayCtx.lineTo(x + side, y + side - cornerLen);
        overlayCtx.stroke();

        // Label above square
        overlayCtx.font = "bold 13px Outfit, sans-serif";
        overlayCtx.fillStyle = boxColor;
        const labelText = lastResult ? `[ LIVE: ${lastResult.verdict} ${(lastResult.confidence * 100).toFixed(0)}% ]` : "[ PLACE FRUIT IN SQUARE ]";
        overlayCtx.fillText(labelText, x + 6, y - 8);
      }

      requestAnimationFrame(drawOverlayLoop);
    }

    // 3. Capture Current Frame Crop & Send to Python AI Backend
    async function captureAndClassify() {
      if (!isStreamActive || isProcessing || video.videoWidth === 0) return;
      isProcessing = true;

      const vw = video.videoWidth;
      const vh = video.videoHeight;
      const side = Math.floor(Math.min(vw, vh) * 0.75);
      const sx = Math.floor((vw - side) / 2);
      const sy = Math.floor((vh - side) / 2);

      // Draw the cropped center area onto the 224x224 processing canvas
      cropCtx.drawImage(video, sx, sy, side, side, 0, 0, 224, 224);

      const base64Img = cropCanvas.toDataURL("image/jpeg", 0.85);

      try {
        const resp = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image: base64Img,
            threshold: certaintyThreshold,
            filename: "live_camera_frame.jpg"
          })
        });

        const data = await resp.json();
        if (data.status === "success") {
          lastResult = data;
          updateDashboard(data);
        }
      } catch (e) {
        console.warn("Prediction request failed:", e);
      } finally {
        isProcessing = false;
      }
    }

    // 4. Update Dashboard UI with Inference Results
    function updateDashboard(data) {
      const box = document.getElementById('verdictBox');
      const vText = document.getElementById('verdictText');
      const vSub = document.getElementById('verdictSub');

      box.className = 'verdict-box ' + data.verdict.toLowerCase();
      vText.innerText = data.verdict;

      const maxPct = (data.confidence * 100).toFixed(1);
      vSub.innerText = `Confidence: ${maxPct}% • Threshold: ${(certaintyThreshold * 100).toFixed(0)}%`;

      document.getElementById('freshPct').innerText = (data.fresh_score * 100).toFixed(1) + '%';
      document.getElementById('rottenPct').innerText = (data.rotten_score * 100).toFixed(1) + '%';
      document.getElementById('freshBar').style.width = (data.fresh_score * 100) + '%';
      document.getElementById('rottenBar').style.width = (data.rotten_score * 100) + '%';

      document.getElementById('latencyStat').innerText = `${data.inference_ms.toFixed(1)} ms`;
    }

    function updateThreshold(val) {
      certaintyThreshold = val / 100.0;
      document.getElementById('thresholdVal').innerText = val + '%';
      if (lastResult) {
        // Recalculate verdict with new threshold
        const maxScore = Math.max(lastResult.fresh_score, lastResult.rotten_score);
        if (maxScore >= certaintyThreshold) {
          lastResult.verdict = lastResult.fresh_score > lastResult.rotten_score ? "FRESH" : "ROTTEN";
        } else {
          lastResult.verdict = "UNCERTAIN";
        }
        updateDashboard(lastResult);
      }
    }

    // 5. Log and Count Current Fruit to Session Table & CSV
    function logCurrentItem() {
      if (!lastResult) return;

      const item = {
        time: new Date().toLocaleTimeString(),
        verdict: lastResult.verdict,
        confidence: (lastResult.confidence * 100).toFixed(1) + '%',
        fresh: (lastResult.fresh_score * 100).toFixed(1) + '%',
        rotten: (lastResult.rotten_score * 100).toFixed(1) + '%',
        latency: lastResult.inference_ms.toFixed(1) + 'ms'
      };

      historyRecords.unshift(item);
      renderHistoryTable();
      document.getElementById('loggedCountStat').innerText = historyRecords.length;

      // Flash button
      const btn = document.getElementById('btnLog');
      const oldHtml = btn.innerHTML;
      btn.innerHTML = "<span>✓ Item Logged!</span>";
      btn.style.background = "#059669";
      setTimeout(() => {
        btn.innerHTML = oldHtml;
        btn.style.background = "";
      }, 1000);
    }

    function renderHistoryTable() {
      const tbody = document.getElementById('historyTbody');
      tbody.innerHTML = historyRecords.map(r => `
        <tr>
          <td>${r.time}</td>
          <td><span class="badge-status ${r.verdict.toLowerCase()}">${r.verdict}</span></td>
          <td><b>${r.confidence}</b></td>
          <td style="color: #34d399;">${r.fresh}</td>
          <td style="color: #f87171;">${r.rotten}</td>
          <td style="color: var(--text-muted);">${r.latency}</td>
        </tr>
      `).join('');
    }

    function downloadCSV() {
      if (historyRecords.length === 0) {
        alert("No items have been logged in this session yet. Press 'Log Current Item' first.");
        return;
      }
      let csvContent = "data:text/csv;charset=utf-8,Time,Verdict,Confidence,FreshScore,RottenScore,Latency\\n";
      historyRecords.forEach(r => {
        csvContent += `${r.time},${r.verdict},${r.confidence},${r.fresh},${r.rotten},${r.latency}\\n`;
      });
      const encodedUri = encodeURI(csvContent);
      const link = document.createElement("a");
      link.setAttribute("href", encodedUri);
      link.setAttribute("download", `fruit_scanner_log_${Date.now()}.csv`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }

    // Spacebar shortcut to log item
    window.addEventListener('keydown', e => {
      if (e.code === 'Space' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'SELECT') {
        e.preventDefault();
        logCurrentItem();
      }
    });

    // Start on load
    startCamera();
  </script>
</body>
</html>
"""

_predictor: Optional[Predictor] = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


class LiveCameraAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler serving the live camera web application and prediction API."""

    def log_message(self, format: str, *args: Any) -> None:
        return

    def send_json(self, status_code: int, data: Dict[str, Any]) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        dist_dir = Path(__file__).resolve().parent.parent / "dist"
        if not dist_dir.exists():
            dist_dir = Path(__file__).resolve().parent / "dist"

        if path in ("/", "/index.html"):
            dist_index = dist_dir / "index.html"
            if dist_index.exists():
                payload = dist_index.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(payload)
                return

            payload = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)

        elif path.startswith("/assets/") or path.startswith("/models/") or path.startswith("/ort/") or path.startswith("/samples/"):
            asset_file = dist_dir / path.lstrip("/")
            if asset_file.exists() and asset_file.is_file():
                if path.endswith(".js") or path.endswith(".mjs"):
                    ctype = "application/javascript"
                elif path.endswith(".css"):
                    ctype = "text/css"
                elif path.endswith(".wasm"):
                    ctype = "application/wasm"
                elif path.endswith(".svg"):
                    ctype = "image/svg+xml"
                elif path.endswith(".png"):
                    ctype = "image/png"
                elif path.endswith(".jpg") or path.endswith(".jpeg"):
                    ctype = "image/jpeg"
                else:
                    ctype = "application/octet-stream"
                payload = asset_file.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(payload)
                return
            else:
                self.send_response(404)
                self.end_headers()

        elif path == "/api/status":
            try:
                pred = get_predictor()
                self.send_json(200, {
                    "status": "ready",
                    "model": "MobileNetV2 (ONNX)",
                    "classes": pred.classes,
                    "default_threshold": pred.default_threshold,
                })
            except Exception as e:
                self.send_json(500, {"status": "error", "message": str(e)})

        elif path == "/classic":
            payload = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/predict":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body.decode("utf-8"))
                raw_b64 = data.get("image", "")
                threshold = float(data.get("threshold", DEFAULT_THRESHOLD))
                filename = data.get("filename", "live_frame.jpg")

                if "," in raw_b64:
                    raw_b64 = raw_b64.split(",", 1)[1]

                img_bytes = base64.b64decode(raw_b64)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if bgr is None:
                    self.send_json(400, {"status": "error", "message": "Invalid image payload."})
                    return

                # Center square crop
                crop, (cx, cy, side) = square(bgr)

                # Fast Inference via OpenCV DNN
                t0 = time.perf_counter()
                predictor = get_predictor()
                probs = predictor.predict(crop)
                dt_ms = (time.perf_counter() - t0) * 1000.0

                fresh_p = float(probs[0])
                rotten_p = float(probs[1])

                winner_idx = int(np.argmax(probs))
                confidence = float(probs[winner_idx])
                winning_class = CLASSES[winner_idx].upper()

                verdict = winning_class if confidence >= threshold else "UNCERTAIN"

                # Log scan event
                log_scan_event(
                    batch_id="LIVE-WEB",
                    result=verdict,
                    fresh_score=fresh_p,
                    rotten_score=rotten_p,
                    threshold=threshold,
                    inference_ms=dt_ms,
                )

                self.send_json(200, {
                    "status": "success",
                    "verdict": verdict,
                    "prediction": verdict.lower(),
                    "confidence": round(confidence, 4),
                    "fresh_score": round(fresh_p, 4),
                    "rotten_score": round(rotten_p, 4),
                    "threshold": threshold,
                    "inference_ms": round(dt_ms, 1),
                })

            except Exception as err:
                self.send_json(500, {"status": "error", "message": str(err)})
        else:
            self.send_response(404)
            self.end_headers()


def launch_web_server(port: int = PORT, open_browser: bool = True) -> None:
    """Start local web server and open live camera GUI in user's default browser."""
    print("=" * 65)
    print("FRUIT FRESHNESS DETECTOR - LIVE CAMERA WEB APP")
    print("=" * 65)
    print(f"Server Port: http://localhost:{port}")
    print(f"Model File:  {MODEL_ONNX_PATH.resolve()}")

    try:
        get_predictor()
        print("[+] MobileNetV2 ONNX model loaded and verified.")
    except Exception as e:
        print(f"[-] Warning: Model load issue: {e}")

    server_address = ("127.0.0.1", port)
    try:
        httpd = ThreadingHTTPServer(server_address, LiveCameraAPIHandler)
    except OSError:
        port = port + 1
        server_address = ("127.0.0.1", port)
        httpd = ThreadingHTTPServer(server_address, LiveCameraAPIHandler)

    url = f"http://localhost:{port}"
    print(f"[+] Live Camera Web App is running at: {url}")
    print("    - Camera stream is active directly inside your browser")
    print("    - Hold an apple, banana, or orange in front of the camera")
    print("    - See live FRESH / ROTTEN diagnosis in real time")
    print("    - Press Spacebar or click 'Log Current Item' to record to history\n")

    if open_browser:
        webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Server stopping...")
    finally:
        httpd.server_close()
        print("[+] Server closed cleanly.")


if __name__ == "__main__":
    launch_web_server(open_browser=True)
