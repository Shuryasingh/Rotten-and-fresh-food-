/**
 * Browser-Native MobileNetV2 ONNX Neural Network Inference Engine
 * Runs the trained PyTorch/Torchvision MobileNetV2 classifier (`fruit.onnx`)
 * directly in Chrome/Edge/Safari/Mobile via WebAssembly (`onnxruntime-web`)
 * with 0 server dependencies — 100% identical accuracy on Vercel and locally.
 */

import * as ort from 'onnxruntime-web/wasm';

// Serve WASM runtime files locally from public/ort/ (bundled into dist/ort/)
// Zero external CDN dependency, COOP/COEP-free single-threaded WASM.
ort.env.wasm.wasmPaths = '/ort/';
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;

const MODEL_URL = '/models/fruit.onnx';
const IMG_SIZE = 224;

// Standard ImageNet normalization matching rotten and fresh fruit dectection/common.py
const IMAGENET_MEAN = [0.485, 0.456, 0.406] as const;
const IMAGENET_STD = [0.229, 0.224, 0.225] as const;

export type OnnxEngineStatus = 'idle' | 'loading' | 'ready' | 'error';

export interface OnnxInferenceOutput {
  freshScore: number;
  rottenScore: number;
  inferenceMs: number;
  fullFrameProbs: [number, number];
  centerFocusProbs: [number, number];
}

let sessionPromise: Promise<ort.InferenceSession> | null = null;
let loadedSession: ort.InferenceSession | null = null;
let engineStatus: OnnxEngineStatus = 'idle';
let engineError: string | null = null;

// Reusable scratch canvas for the inner center-focus crop (avoids GC churn at 3 FPS)
let scratchCanvas: HTMLCanvasElement | null = null;

function getScratchCanvas(): HTMLCanvasElement {
  if (!scratchCanvas) {
    scratchCanvas = document.createElement('canvas');
    scratchCanvas.width = IMG_SIZE;
    scratchCanvas.height = IMG_SIZE;
  }
  return scratchCanvas;
}

/**
 * Eagerly initializes (and caches) the MobileNetV2 ONNX InferenceSession.
 */
export async function initOnnxEngine(): Promise<ort.InferenceSession> {
  if (loadedSession) {
    return loadedSession;
  }
  if (sessionPromise) {
    return sessionPromise;
  }

  engineStatus = 'loading';
  engineError = null;

  sessionPromise = (async () => {
    try {
      const session = await ort.InferenceSession.create(MODEL_URL, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      loadedSession = session;
      engineStatus = 'ready';
      return session;
    } catch (err) {
      engineStatus = 'error';
      engineError = err instanceof Error ? err.message : String(err);
      sessionPromise = null;
      throw err;
    }
  })();

  return sessionPromise;
}

export function getOnnxEngineState(): { status: OnnxEngineStatus; error: string | null } {
  return { status: engineStatus, error: engineError };
}

/**
 * Converts 224x224 Canvas ImageData into NCHW Float32 Tensor [1, 3, 224, 224]
 * matching OpenCV/PyTorch MobileNetV2 ImageNet standardization bit-for-bit.
 */
function imageDataToTensor(imgData: ImageData): ort.Tensor {
  const { data } = imgData;
  const area = IMG_SIZE * IMG_SIZE;
  const floatData = new Float32Array(3 * area);

  for (let i = 0; i < area; i++) {
    const p = i * 4;
    const r = data[p] / 255.0;
    const g = data[p + 1] / 255.0;
    const b = data[p + 2] / 255.0;

    floatData[i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    floatData[area + i] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
    floatData[2 * area + i] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
  }

  return new ort.Tensor('float32', floatData, [1, 3, IMG_SIZE, IMG_SIZE]);
}

function softmax2(logits: ArrayLike<number>): [number, number] {
  const l0 = Number(logits[0]);
  const l1 = Number(logits[1]);
  const maxLogit = Math.max(l0, l1);
  const e0 = Math.exp(l0 - maxLogit);
  const e1 = Math.exp(l1 - maxLogit);
  const sum = e0 + e1;
  return [e0 / sum, e1 / sum];
}

async function runSingleTensor(session: ort.InferenceSession, imgData: ImageData): Promise<[number, number]> {
  const tensor = imageDataToTensor(imgData);
  const inputName = session.inputNames[0] || 'input';
  const outputName = session.outputNames[0] || 'output';
  const results = await session.run({ [inputName]: tensor });
  const outputTensor = results[outputName];
  return softmax2(outputTensor.data as Float32Array);
}

/**
 * Executes dual-scale ONNX neural network inference on a 224x224 specimen canvas:
 * 1) Full inspection square (preserves stem/crown/edge lesions & overall silhouette)
 * 2) Tight inner 74% focus crop (strips surrounding indoor wall/hand clutter)
 */
export async function predictWithOnnx(
  canvas: HTMLCanvasElement,
  enableMultiCrop: boolean = true
): Promise<OnnxInferenceOutput> {
  const t0 = performance.now();
  const session = await initOnnxEngine();

  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) {
    throw new Error('Could not read 2D canvas context.');
  }

  const fullImgData = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
  const fullProbs = await runSingleTensor(session, fullImgData);

  let centerProbs: [number, number] = fullProbs;
  let freshScore = fullProbs[0];
  let rottenScore = fullProbs[1];

  if (enableMultiCrop) {
    // Inner 74% crop focused tightly on center fruit body
    const margin = Math.round(IMG_SIZE * 0.13); // 29px margin -> 166x166 center
    const cropSide = IMG_SIZE - margin * 2;
    const sCanvas = getScratchCanvas();
    const sCtx = sCanvas.getContext('2d', { willReadFrequently: true });
    if (sCtx) {
      sCtx.drawImage(canvas, margin, margin, cropSide, cropSide, 0, 0, IMG_SIZE, IMG_SIZE);
      const centerImgData = sCtx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
      centerProbs = await runSingleTensor(session, centerImgData);

      // Weighted ensemble favoring full view for boundary lesions + center crop for clutter immunity
      freshScore = 0.62 * fullProbs[0] + 0.38 * centerProbs[0];
      rottenScore = 0.62 * fullProbs[1] + 0.38 * centerProbs[1];

      // Rotten recall safety: if either view strongly spots localized rot (>= 0.90)
      // while the other view also leans non-trivial rot (>= 0.35), preserve high rot sensitivity.
      if ((fullProbs[1] >= 0.90 && centerProbs[1] >= 0.35) || (centerProbs[1] >= 0.90 && fullProbs[1] >= 0.35)) {
        rottenScore = Math.max(rottenScore, 0.72 * Math.max(fullProbs[1], centerProbs[1]) + 0.28 * Math.min(fullProbs[1], centerProbs[1]));
        freshScore = 1 - rottenScore;
      }
    }
  }

  const inferenceMs = Math.round((performance.now() - t0) * 10) / 10;

  return {
    freshScore: Math.round(freshScore * 10000) / 10000,
    rottenScore: Math.round(rottenScore * 10000) / 10000,
    inferenceMs,
    fullFrameProbs: fullProbs,
    centerFocusProbs: centerProbs,
  };
}
