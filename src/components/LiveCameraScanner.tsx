import React, { useRef, useState, useEffect, useCallback } from 'react';
import {
  Camera,
  Play,
  Pause,
  Save,
  CheckCircle,
  AlertTriangle,
  HelpCircle,
  Sliders,
  Square,
  Upload,
  Sun,
  Crosshair,
  Maximize2,
  Sparkles,
  Trash2,
  Activity,
} from 'lucide-react';
import { analyzeFruitFrame, normalizeFruitVariety } from '../utils/fruitVision';
import { initOnnxEngine, predictWithOnnx, OnnxEngineStatus } from '../utils/onnxFruitEngine';

interface PredictionResponse {
  status: string;
  prediction?: 'fresh' | 'rotten' | 'uncertain';
  verdict?: string;
  confidence: number;
  fresh_score: number;
  rotten_score: number;
  inference_ms: number;
}

interface SampleSpecimen {
  id: string;
  name: string;
  fruitType: string;
  expected: 'fresh' | 'rotten';
  imageUrl: string;
  description: string;
}

interface ScanRecord {
  id: string;
  timestamp: string;
  fruit: string;
  verdict: 'fresh' | 'rotten' | 'uncertain';
  confidence: number;
  thumbUrl: string;
}

const SAMPLE_PRESETS: SampleSpecimen[] = [
  {
    id: 's-apple-f',
    name: 'Fresh Fuji Red Apple',
    fruitType: 'apple',
    expected: 'fresh',
    imageUrl: '/samples/fresh-apple.jpg',
    description: 'Firm red epidermis, unbroken cellular wax cuticle.',
  },
  {
    id: 's-apple-r',
    name: 'Rotten Bruised Apple',
    fruitType: 'apple',
    expected: 'rotten',
    imageUrl: '/samples/rotten-apple.jpg',
    description: 'Sunken brown bitter rot lesion & soft necrosis.',
  },
  {
    id: 's-banana-f',
    name: 'Fresh Yellow Banana',
    fruitType: 'banana',
    expected: 'fresh',
    imageUrl: '/samples/fresh-banana.jpg',
    description: 'Uniform golden carotenoid peel, firm cellular wall.',
  },
  {
    id: 's-banana-r',
    name: 'Rotten Necrotic Banana',
    fruitType: 'banana',
    expected: 'rotten',
    imageUrl: '/samples/rotten-banana.jpg',
    description: 'Blackened epidermal rot & crown liquefaction.',
  },
  {
    id: 's-orange-f',
    name: 'Fresh Mandarin Citrus',
    fruitType: 'orange',
    expected: 'fresh',
    imageUrl: '/samples/fresh-orange.jpg',
    description: 'Intact oil glands, hydrated leaf & vibrant flavedo.',
  },
  {
    id: 's-orange-r',
    name: 'Rotten Moldy Citrus',
    fruitType: 'orange',
    expected: 'rotten',
    imageUrl: '/samples/rotten-orange.jpg',
    description: 'Penicillium digitatum blue-green mold sporulation.',
  },
  {
    id: 's-green-apple-f',
    name: 'Fresh Granny Smith',
    fruitType: 'apple',
    expected: 'fresh',
    imageUrl: '/samples/fresh-green-apple.jpg',
    description: 'Uniform chlorophyll peel with unbroken turgor.',
  },
  {
    id: 's-apple-ox-r',
    name: 'Rotten Oxidized Apple',
    fruitType: 'apple',
    expected: 'rotten',
    imageUrl: '/samples/rotten-oxidized-apple.jpg',
    description: 'Severe PPO enzymatic browning & epidermal collapse.',
  },
  {
    id: 's-mango-f',
    name: 'Fresh Alphonso Mango',
    fruitType: 'mango',
    expected: 'fresh',
    imageUrl: 'https://images.unsplash.com/photo-1553279768-865429fa0078?q=80&w=640&auto=format&fit=crop',
    description: 'Vibrant golden pericarp, firm cuticle, unbroken bloom.',
  },
  {
    id: 's-strawberry-f',
    name: 'Fresh Ruby Strawberry',
    fruitType: 'strawberry',
    expected: 'fresh',
    imageUrl: 'https://images.unsplash.com/photo-1464965911861-746a04b4bca6?q=80&w=640&auto=format&fit=crop',
    description: 'Glossy red drupelets, hydrated bright green calyx.',
  },
];

export const LiveCameraScanner: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Mode: 'camera' | 'sample' | 'upload'
  const [activeInputMode, setActiveInputMode] = useState<'camera' | 'sample' | 'upload'>('camera');
  const [activeSample, setActiveSample] = useState<SampleSpecimen | null>(null);
  const [uploadedImageSrc, setUploadedImageSrc] = useState<string | null>(null);

  // Camera State
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const [selectedFruit, setSelectedFruit] = useState<string>('all');
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isPaused, setIsPaused] = useState(false);

  // Sample Space Calibration
  const [targetBoxSize, setTargetBoxSize] = useState<number>(75); // 50% - 90%
  const [reticleStyle, setReticleStyle] = useState<'square' | 'crosshair' | 'corners'>('square');
  const [luminosity, setLuminosity] = useState<number>(120);

  // Inference State
  const [threshold, setThreshold] = useState(80);
  const [prediction, setPrediction] = useState<'fresh' | 'rotten' | 'uncertain' | 'idle' | 'initializing'>('idle');
  const [freshScore, setFreshScore] = useState(0);
  const [rottenScore, setRottenScore] = useState(0);
  const [inferenceMs, setInferenceMs] = useState<number | null>(null);
  const [fps, setFps] = useState(2.5);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [onnxStatus, setOnnxStatus] = useState<OnnxEngineStatus>('loading');
  const [detectedBiomarker, setDetectedBiomarker] = useState<string>('Unbroken Epicuticle & Healthy Cellular Turgor');
  const [necroticPercent, setNecroticPercent] = useState<number>(0);
  const [healthyPercent, setHealthyPercent] = useState<number>(95);
  const [engineType, setEngineType] = useState<string>('MobileNetV2 ONNX WASM (Loading...)');

  // Temporal EMA smoothing ref for flicker-free live webcam verdicts
  const emaScoresRef = useRef<{ fresh: number; rotten: number } | null>(null);
  const isInferringRef = useRef<boolean>(false);

  // Logged Scan History
  const [scanHistory, setScanHistory] = useState<ScanRecord[]>([]);

  // Preload MobileNetV2 ONNX model immediately on mount so Vercel & local startup is instant
  useEffect(() => {
    let mounted = true;
    setOnnxStatus('loading');
    initOnnxEngine()
      .then(() => {
        if (!mounted) return;
        setOnnxStatus('ready');
        setEngineType('MobileNetV2 ONNX Neural Net (Browser WASM)');
      })
      .catch((err) => {
        console.warn('ONNX initialization fallback active:', err);
        if (!mounted) return;
        setOnnxStatus('error');
        setEngineType('Bio-Optical Fallback Engine');
      });
    return () => {
      mounted = false;
    };
  }, []);

  // Re-evaluate verdict immediately when threshold slider moves
  useEffect(() => {
    if (prediction === 'fresh' || prediction === 'rotten' || prediction === 'uncertain') {
      const thr = threshold / 100;
      if (freshScore >= thr) setPrediction('fresh');
      else if (rottenScore >= thr) setPrediction('rotten');
      else setPrediction('uncertain');
    }
  }, [threshold, freshScore, rottenScore, prediction]);

  // Enumerate video devices
  useEffect(() => {
    async function getDevices() {
      try {
        const devList = await navigator.mediaDevices.enumerateDevices();
        const videoInputs = devList.filter((d) => d.kind === 'videoinput');
        setDevices(videoInputs);
        if (videoInputs.length > 0 && !selectedDeviceId) {
          setSelectedDeviceId(videoInputs[0].deviceId);
        }
      } catch (err) {
        console.warn('Could not enumerate media devices:', err);
      }
    }
    getDevices();
  }, [selectedDeviceId]);

  // Start webcam
  const startCamera = useCallback(async (deviceId?: string) => {
    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }

      const constraints: MediaStreamConstraints = {
        video: {
          deviceId: deviceId ? { exact: deviceId } : undefined,
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      emaScoresRef.current = null;
      setIsCameraActive(true);
      setIsPaused(false);
      setActiveInputMode('camera');
      setPrediction('initializing');
    } catch (err) {
      console.error('Camera access error:', err);
      setIsCameraActive(false);
    }
  }, []);

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    emaScoresRef.current = null;
    setIsCameraActive(false);
    setPrediction('idle');
  };

  const togglePause = () => {
    if (!videoRef.current) return;
    if (isPaused) {
      videoRef.current.play();
      setIsPaused(false);
    } else {
      videoRef.current.pause();
      setIsPaused(true);
    }
  };

  // Real-Time Hybrid Neural + Presence-Gated Inference Executor (Works 100% on Vercel & Local)
  const executeInference = useCallback(
    async (canvas: HTMLCanvasElement, fruitTag: string, isLiveStream: boolean = false) => {
      if (isInferringRef.current && isLiveStream) return;
      isInferringRef.current = true;
      const startTime = performance.now();

      try {
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        if (!ctx) return;

        let imgData: ImageData | null = null;
        try {
          imgData = ctx.getImageData(0, 0, 224, 224);
        } catch (err) {
          console.warn('Canvas pixel extraction notice:', err);
        }

        if (!imgData) return;

        const thr = threshold / 100;
        const variety = normalizeFruitVariety(fruitTag);

        // 1. Specimen Presence & Lighting Gate (blocks empty walls / bare user face in live camera)
        const presence = analyzeFruitFrame(imgData, fruitTag, thr);
        setLuminosity(presence.luminosity);

        if (isLiveStream && !presence.fruitDetected) {
          emaScoresRef.current = null;
          setPrediction('initializing');
          setFreshScore(0.0);
          setRottenScore(0.0);
          setDetectedBiomarker(presence.primaryBiomarker);
          setNecroticPercent(0);
          setHealthyPercent(0);
          return;
        }

        // 2. Run Real MobileNetV2 ONNX Neural Network in-browser (primary accuracy engine)
        let rawFresh = presence.fresh_score;
        let rawRotten = presence.rotten_score;
        let latencyMs = Math.round((performance.now() - startTime) * 10) / 10;
        let activeEngine = 'Bio-Optical Spectral Fallback';
        let neuralSuccess = false;

        try {
          const onnxOut = await predictWithOnnx(canvas, true);
          rawFresh = onnxOut.freshScore;
          rawRotten = onnxOut.rottenScore;
          latencyMs = onnxOut.inferenceMs;
          setOnnxStatus('ready');
          setBackendOnline(false);
          activeEngine = 'MobileNetV2 ONNX Neural Net (Browser WASM)';
          neuralSuccess = true;
        } catch (onnxErr) {
          // Optional localhost Python fallback ONLY if running locally alongside web_app.py
          const isLocalHost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
          if (isLocalHost && window.location.protocol === 'http:') {
            try {
              const apiUrl = window.location.port === '8080' ? '/api/predict' : 'http://localhost:8080/api/predict';
              const base64Img = canvas.toDataURL('image/jpeg', 0.85);
              const resp = await fetch(apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  image: base64Img,
                  threshold: thr,
                  filename: `${fruitTag}_scan.jpg`,
                }),
              });
              if (resp.ok) {
                const data: PredictionResponse = await resp.json();
                setBackendOnline(true);
                rawFresh = data.fresh_score;
                rawRotten = data.rotten_score;
                latencyMs = data.inference_ms;
                activeEngine = 'MobileNetV2 ONNX (Python Backend)';
                neuralSuccess = true;
              }
            } catch {
              setBackendOnline(false);
            }
          } else {
            console.warn('In-browser ONNX inference unavailable, using spectral fallback:', onnxErr);
          }
        }

        // 3. Out-of-distribution specialty guard for dark purple grapes & dark Hass avocado
        // (MobileNetV2 was trained on Apple/Banana/Orange; purple grape clusters look brown/dark without spectral guidance)
        const isDarkSpecialty =
          variety === 'grapes' ||
          variety === 'avocado' ||
          (variety === 'all' && presence.dominantPurpleGrapeRatio > 0.28);

        if (neuralSuccess && isDarkSpecialty) {
          rawFresh = Math.round((0.25 * rawFresh + 0.75 * presence.fresh_score) * 10000) / 10000;
          rawRotten = Math.round((1 - rawFresh) * 10000) / 10000;
          activeEngine = 'Hybrid MobileNetV2 + Spectral Cultivar Model';
        }

        // 4. Temporal EMA smoothing on live camera feed so verdicts don't flicker across frames
        let finalFresh = rawFresh;
        let finalRotten = rawRotten;
        if (isLiveStream) {
          if (emaScoresRef.current) {
            const alpha = 0.58;
            finalFresh = alpha * rawFresh + (1 - alpha) * emaScoresRef.current.fresh;
            finalRotten = 1 - finalFresh;
          }
          emaScoresRef.current = { fresh: finalFresh, rotten: finalRotten };
        } else {
          emaScoresRef.current = null;
        }

        finalFresh = Math.round(finalFresh * 10000) / 10000;
        finalRotten = Math.round(finalRotten * 10000) / 10000;

        // 5. Strict 3-State Certainty Decision
        let finalVerdict: 'fresh' | 'rotten' | 'uncertain' = 'uncertain';
        if (finalFresh >= thr) finalVerdict = 'fresh';
        else if (finalRotten >= thr) finalVerdict = 'rotten';

        // Align biomarker telemetry cleanly with the neural network diagnosis
        const telemetry = analyzeFruitFrame(imgData, fruitTag, thr, {
          prediction: finalVerdict,
          freshScore: finalFresh,
          rottenScore: finalRotten,
        });

        setPrediction(finalVerdict);
        setFreshScore(finalFresh);
        setRottenScore(finalRotten);
        setInferenceMs(latencyMs);
        setEngineType(activeEngine);
        setDetectedBiomarker(telemetry.primaryBiomarker);
        setNecroticPercent(telemetry.necroticAreaPercent);
        setHealthyPercent(telemetry.healthyChromaPercent);

        const elapsed = performance.now() - startTime;
        setFps(Math.round((1000 / Math.max(elapsed, 200)) * 10) / 10);
      } finally {
        isInferringRef.current = false;
      }
    },
    [threshold]
  );

  // Periodic Live Camera Loop
  useEffect(() => {
    if (activeInputMode !== 'camera' || !isCameraActive || isPaused) return;

    let isCancelled = false;
    const interval = setInterval(async () => {
      if (isCancelled || !videoRef.current || !canvasRef.current) return;

      const video = videoRef.current;
      if (video.readyState < 2 || video.videoWidth === 0) return;

      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) return;

      // Extract center square crop matching the cyan target reticle
      const vw = video.videoWidth;
      const vh = video.videoHeight;
      const size = Math.min(vw, vh) * (targetBoxSize / 100);
      const sx = (vw - size) / 2;
      const sy = (vh - size) / 2;

      canvas.width = 224;
      canvas.height = 224;
      ctx.drawImage(video, sx, sy, size, size, 0, 0, 224, 224);

      await executeInference(canvas, selectedFruit, true);
    }, 350);

    return () => {
      isCancelled = true;
      clearInterval(interval);
    };
  }, [activeInputMode, isCameraActive, isPaused, targetBoxSize, selectedFruit, executeInference]);

  // Handle Preset Sample Selection (runs genuine ONNX inference, no mocked results)
  const handleSelectSample = (sample: SampleSpecimen) => {
    emaScoresRef.current = null;
    setActiveInputMode('sample');
    setActiveSample(sample);
    setSelectedFruit(sample.fruitType);
    setPrediction('initializing');

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = sample.imageUrl;
    img.onload = () => {
      if (!canvasRef.current) return;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) return;

      canvas.width = 224;
      canvas.height = 224;
      const minDim = Math.min(img.width, img.height);
      const sx = (img.width - minDim) / 2;
      const sy = (img.height - minDim) / 2;
      ctx.drawImage(img, sx, sy, minDim, minDim, 0, 0, 224, 224);

      executeInference(canvas, sample.fruitType, false);
    };
  };

  // Handle Upload Custom Photo
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    emaScoresRef.current = null;
    const reader = new FileReader();
    reader.onload = (event) => {
      const result = event.target?.result as string;
      setUploadedImageSrc(result);
      setActiveInputMode('upload');
      setPrediction('initializing');

      const img = new Image();
      img.src = result;
      img.onload = () => {
        if (!canvasRef.current) return;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        if (!ctx) return;

        canvas.width = 224;
        canvas.height = 224;
        const minDim = Math.min(img.width, img.height);
        const sx = (img.width - minDim) / 2;
        const sy = (img.height - minDim) / 2;
        ctx.drawImage(img, sx, sy, minDim, minDim, 0, 0, 224, 224);

        executeInference(canvas, selectedFruit, false);
      };
    };
    reader.readAsDataURL(file);
  };

  // Log Current Item Snapshot
  const handleLogItem = () => {
    if (!canvasRef.current) return;
    const thumbUrl = canvasRef.current.toDataURL('image/jpeg', 0.6);
    const newRecord: ScanRecord = {
      id: `SCAN-${Date.now().toString().slice(-4)}`,
      timestamp: new Date().toLocaleTimeString(),
      fruit: selectedFruit.toUpperCase(),
      verdict: prediction === 'initializing' || prediction === 'idle' ? 'uncertain' : prediction,
      confidence: prediction === 'rotten' ? rottenScore : freshScore,
      thumbUrl,
    };
    setScanHistory((prev) => [newRecord, ...prev.slice(0, 7)]);
  };

  // Fruit Guidance Text
  const getFruitGuidance = () => {
    switch (selectedFruit) {
      case 'mango':
        return '🥭 MANGO MODE: Point camera at stem-end & cheeks. Scanning for anthracnose black spots & soft lesions.';
      case 'grapes':
        return '🍇 GRAPES MODE: Hold grape cluster in square. Scanning for Botrytis gray mold, slip-skin & leaky berries.';
      case 'strawberry':
        return '🍓 STRAWBERRY MODE: Scanning for cottony Rhizopus mycelium, water-soaked patches & calyx rot.';
      case 'avocado':
        return '🥑 AVOCADO MODE: Scanning for stem-button cavitation, squishy bruises & vascular decay.';
      case 'apple':
        return '🍎 APPLE MODE: Scanning for sunken bitter rot, soft brown bruising & lenticel breakdown.';
      case 'banana':
        return '🍌 BANANA MODE: Scanning crown rot & fungal breakdown vs harmless sugar freckles.';
      case 'orange':
        return '🍊 CITRUS MODE: Scanning for blue-green Penicillium mold spores & softening pericarp.';
      default:
        return 'Hold any fruit (mango, grapes, apple, banana, orange, berry, avocado) inside the inspection square.';
    }
  };

  return (
    <section id="scanner" className="relative bg-[#07090e] text-[#f8fafc] py-24 px-6 md:px-12 lg:px-16 border-t border-white/10">
      <div className="max-w-[1360px] mx-auto space-y-10">
        {/* Section Top Header */}
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6 border-b border-white/10 pb-6">
          <div className="space-y-1">
            <div className="flex items-center gap-3">
              <span className="text-3xl">🥭</span>
              <h2 className="font-serif text-3xl md:text-4xl text-white tracking-tight">
                Live Camera Test Lab & Sample Space
              </h2>
            </div>
            <p className="text-xs sm:text-sm text-[#94a3b8] font-sans">
              Real-time MobileNetV2 ONNX camera test sandbox with target calibration, instant sample specimens, and lighting analytics.
            </p>
          </div>

          {/* Test Input Mode Switcher */}
          <div className="flex flex-wrap items-center gap-3">
            <div className="bg-[#0f141f] border border-white/10 rounded-lg p-1 flex items-center gap-1 text-xs font-medium">
              <button
                onClick={() => {
                  setActiveInputMode('camera');
                  if (!isCameraActive) startCamera(selectedDeviceId);
                }}
                className={`px-3 py-1.5 rounded transition-all inline-flex items-center gap-1.5 ${
                  activeInputMode === 'camera'
                    ? 'bg-[#06b6d4] text-[#07090e] font-semibold shadow'
                    : 'text-[#94a3b8] hover:text-white'
                }`}
              >
                <Camera className="w-3.5 h-3.5" />
                <span>Live Webcam</span>
              </button>

              <button
                onClick={() => {
                  setActiveInputMode('sample');
                  if (!activeSample) handleSelectSample(SAMPLE_PRESETS[0]);
                }}
                className={`px-3 py-1.5 rounded transition-all inline-flex items-center gap-1.5 ${
                  activeInputMode === 'sample'
                    ? 'bg-[#10b981] text-[#07090e] font-semibold shadow'
                    : 'text-[#94a3b8] hover:text-white'
                }`}
              >
                <Sparkles className="w-3.5 h-3.5" />
                <span>Sample Space Presets</span>
              </button>

              <button
                onClick={() => fileInputRef.current?.click()}
                className={`px-3 py-1.5 rounded transition-all inline-flex items-center gap-1.5 ${
                  activeInputMode === 'upload'
                    ? 'bg-[#f59e0b] text-[#07090e] font-semibold shadow'
                    : 'text-[#94a3b8] hover:text-white'
                }`}
              >
                <Upload className="w-3.5 h-3.5" />
                <span>Upload Photo</span>
              </button>

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileUpload}
                className="hidden"
              />
            </div>

            <div className="inline-flex items-center gap-2 bg-[#10b981]/15 text-[#10b981] border border-[#10b981]/30 px-3.5 py-1.5 rounded-full text-xs font-sans font-medium uppercase tracking-wider">
              <span className={`w-2 h-2 rounded-full animate-pulse ${onnxStatus === 'ready' || backendOnline ? 'bg-[#10b981]' : 'bg-[#06b6d4]'}`} />
              <span>
                {backendOnline
                  ? 'MOBILENETV2 ONNX (PORT 8080)'
                  : onnxStatus === 'ready'
                  ? 'MOBILENETV2 ONNX READY (IN-BROWSER)'
                  : onnxStatus === 'loading'
                  ? 'LOADING NEURAL MODEL...'
                  : 'SPECTRAL VISION ACTIVE'}
              </span>
            </div>
          </div>
        </div>

        {/* SAMPLE PRESETS SHELF (Quick 1-Click Live Test) */}
        <div className="bg-[#0f141f] border border-white/10 rounded-xl p-5 space-y-3 shadow-xl">
          <div className="flex items-center justify-between text-xs text-[#94a3b8]">
            <span className="font-mono uppercase tracking-wider text-white flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-emerald-400" />
              <span>Quick Test Sample Space (Click Any Fruit to Feed Live Scanner)</span>
            </span>
            <span className="text-[11px] text-[#64748b]">10 Instant Benchmark Specimens</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3">
            {SAMPLE_PRESETS.map((sample) => {
              const isSelected = activeInputMode === 'sample' && activeSample?.id === sample.id;
              return (
                <button
                  key={sample.id}
                  onClick={() => handleSelectSample(sample)}
                  className={`text-left p-2.5 rounded-lg border transition-all flex flex-col justify-between group ${
                    isSelected
                      ? 'border-[#10b981] bg-[#10b981]/10 shadow-[0_0_12px_rgba(16,185,129,0.25)]'
                      : 'border-white/5 bg-[#141b29] hover:border-white/20'
                  }`}
                >
                  <div className="relative aspect-square w-full rounded overflow-hidden mb-2 bg-black/40">
                    <img
                      src={sample.imageUrl}
                      alt={sample.name}
                      className="w-full h-full object-cover transition-transform group-hover:scale-105"
                    />
                    <span
                      className={`absolute top-1.5 right-1.5 text-[9px] font-mono uppercase px-1.5 py-0.5 rounded font-bold ${
                        sample.expected === 'fresh'
                          ? 'bg-[#10b981] text-[#07090e]'
                          : 'bg-[#ef4444] text-white'
                      }`}
                    >
                      {sample.expected}
                    </span>
                  </div>
                  <div>
                    <div className="text-xs font-medium text-white truncate">{sample.name}</div>
                    <div className="text-[10px] text-[#94a3b8] truncate">{sample.description}</div>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Live Test Arena Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left Column: Viewfinder & Target Reticle */}
          <div className="lg:col-span-7 bg-[#0f141f] border border-white/10 rounded-xl p-6 space-y-5 shadow-2xl">
            {/* Viewfinder Top Bar */}
            <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-[#94a3b8]">
              <div className="flex items-center gap-2 font-medium text-white">
                <Camera className="w-4 h-4 text-[#06b6d4]" />
                <span>
                  {activeInputMode === 'camera' && 'Live Camera Viewfinder'}
                  {activeInputMode === 'sample' && `Sample Space Specimen: ${activeSample?.name}`}
                  {activeInputMode === 'upload' && 'Uploaded Custom Specimen'}
                </span>
              </div>

              {/* Real-time Environmental Luminosity Meter */}
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-1.5 bg-[#1e293b] px-2.5 py-1 rounded text-[11px] font-mono">
                  <Sun className="w-3.5 h-3.5 text-amber-400" />
                  <span>Lux: {luminosity}</span>
                  <span
                    className={`ml-1 px-1 rounded text-[9px] font-bold ${
                      luminosity > 220
                        ? 'text-red-400'
                        : luminosity < 60
                        ? 'text-amber-400'
                        : 'text-emerald-400'
                    }`}
                  >
                    {luminosity > 220 ? 'OVEREXPOSED' : luminosity < 60 ? 'DIM' : 'OPTIMAL'}
                  </span>
                </div>
                <div className="font-mono text-white/80">Inference: ~{fps} FPS</div>
              </div>
            </div>

            {/* Viewfinder Canvas Frame */}
            <div className="relative aspect-[4/3] bg-[#05070a] rounded-lg overflow-hidden flex items-center justify-center border border-white/5 select-none">
              {/* Mode 1: Live Webcam Feed */}
              {activeInputMode === 'camera' && (
                <>
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className={`w-full h-full object-cover ${isCameraActive ? 'block' : 'hidden'}`}
                  />
                  {!isCameraActive && (
                    <div className="text-center space-y-4 p-8">
                      <div className="w-16 h-16 rounded-full bg-white/5 border border-white/10 flex items-center justify-center mx-auto text-[#06b6d4]">
                        <Camera className="w-8 h-8" />
                      </div>
                      <div>
                        <h3 className="font-serif text-xl text-white">Webcam Offline</h3>
                        <p className="text-xs text-[#94a3b8] max-w-xs mx-auto mt-1">
                          Activate webcam for real-time live fruit condition classification.
                        </p>
                      </div>
                      <button
                        onClick={() => startCamera(selectedDeviceId)}
                        className="premium-button px-6 py-2.5 bg-[#06b6d4] text-[#07090e] font-semibold text-xs tracking-wider uppercase rounded shadow-lg hover:bg-[#22d3ee] transition-all"
                      >
                        Enable Camera Feed
                      </button>
                    </div>
                  )}
                </>
              )}

              {/* Mode 2: Sample Preset Image */}
              {activeInputMode === 'sample' && activeSample && (
                <div className="relative w-full h-full">
                  <img
                    src={activeSample.imageUrl}
                    alt={activeSample.name}
                    className="w-full h-full object-contain bg-[#05070a]"
                  />
                  <div className="absolute top-4 left-4 bg-black/80 backdrop-blur-md px-3 py-1.5 rounded border border-white/10 text-xs">
                    <span className="text-[#10b981] font-semibold">Testing Preset: </span>
                    <span className="text-white">{activeSample.name}</span>
                  </div>
                </div>
              )}

              {/* Mode 3: Uploaded Custom Image */}
              {activeInputMode === 'upload' && uploadedImageSrc && (
                <div className="relative w-full h-full">
                  <img
                    src={uploadedImageSrc}
                    alt="Custom Uploaded Fruit"
                    className="w-full h-full object-contain bg-[#05070a]"
                  />
                </div>
              )}

              {/* Interactive Reticle Target Square Overlay */}
              <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
                <div
                  className={`relative border-2 rounded flex items-start justify-center pt-2 transition-all duration-300 ${
                    prediction === 'fresh'
                      ? 'border-[#10b981] shadow-[0_0_24px_rgba(16,185,129,0.5)]'
                      : prediction === 'rotten'
                      ? 'border-[#ef4444] shadow-[0_0_24px_rgba(239,68,68,0.55)]'
                      : prediction === 'uncertain'
                      ? 'border-[#f59e0b] shadow-[0_0_20px_rgba(245,158,11,0.4)]'
                      : 'border-[#06b6d4] shadow-[0_0_20px_rgba(6,182,212,0.35)]'
                  } ${reticleStyle === 'crosshair' ? 'border-dashed' : ''}`}
                  style={{
                    width: `${targetBoxSize}%`,
                    height: `${targetBoxSize}%`,
                  }}
                >
                  <span
                    className={`bg-[#0f141f]/95 text-[10px] font-mono px-2.5 py-1 rounded tracking-widest uppercase border shadow ${
                      prediction === 'fresh'
                        ? 'text-[#10b981] border-[#10b981]/60'
                        : prediction === 'rotten'
                        ? 'text-[#ef4444] border-[#ef4444]/60'
                        : prediction === 'uncertain'
                        ? 'text-[#f59e0b] border-[#f59e0b]/60'
                        : 'text-[#06b6d4] border-[#06b6d4]/40'
                    }`}
                  >
                    {prediction === 'fresh'
                      ? `[ FRESH • ${(freshScore * 100).toFixed(1)}% ]`
                      : prediction === 'rotten'
                      ? `[ ROTTEN • ${(rottenScore * 100).toFixed(1)}% ]`
                      : prediction === 'uncertain'
                      ? `[ UNCERTAIN • CHECK LIGHTING ]`
                      : `[ PLACE ${selectedFruit.toUpperCase()} IN SQUARE ]`}
                  </span>

                  {reticleStyle === 'crosshair' && (
                    <div className="absolute inset-0 flex items-center justify-center opacity-40">
                      <div className="w-full h-[1px] bg-[#06b6d4]" />
                      <div className="h-full w-[1px] bg-[#06b6d4] absolute" />
                    </div>
                  )}

                  {reticleStyle === 'corners' && (
                    <>
                      <div className="absolute top-0 left-0 w-3 h-3 border-t-2 border-l-2 border-emerald-400" />
                      <div className="absolute top-0 right-0 w-3 h-3 border-t-2 border-r-2 border-emerald-400" />
                      <div className="absolute bottom-0 left-0 w-3 h-3 border-b-2 border-l-2 border-emerald-400" />
                      <div className="absolute bottom-0 right-0 w-3 h-3 border-b-2 border-r-2 border-emerald-400" />
                    </>
                  )}
                </div>
              </div>

              {/* Hidden 224x224 Canvas for Model Processing */}
              <canvas ref={canvasRef} className="hidden" width={224} height={224} />
            </div>

            {/* Target Reticle Calibration Toolbar (NEW) */}
            <div className="bg-[#141b29] p-3.5 rounded-lg border border-white/5 flex flex-wrap items-center justify-between gap-4 text-xs">
              {/* Target Square Size Slider */}
              <div className="flex items-center gap-3">
                <Maximize2 className="w-4 h-4 text-[#06b6d4]" />
                <span className="text-[#94a3b8]">Target Size:</span>
                <input
                  type="range"
                  min="50"
                  max="90"
                  step="5"
                  value={targetBoxSize}
                  onChange={(e) => setTargetBoxSize(Number(e.target.value))}
                  className="w-24 accent-[#06b6d4] bg-[#1e293b] h-1.5 rounded cursor-pointer"
                />
                <span className="font-mono text-white">{targetBoxSize}%</span>
              </div>

              {/* Reticle Style Selector */}
              <div className="flex items-center gap-2">
                <Crosshair className="w-4 h-4 text-[#06b6d4]" />
                <span className="text-[#94a3b8]">Reticle:</span>
                <button
                  onClick={() => setReticleStyle('square')}
                  className={`px-2 py-1 rounded text-[11px] ${
                    reticleStyle === 'square' ? 'bg-[#06b6d4] text-black font-semibold' : 'bg-[#1e293b] text-white'
                  }`}
                >
                  Solid
                </button>
                <button
                  onClick={() => setReticleStyle('crosshair')}
                  className={`px-2 py-1 rounded text-[11px] ${
                    reticleStyle === 'crosshair' ? 'bg-[#06b6d4] text-black font-semibold' : 'bg-[#1e293b] text-white'
                  }`}
                >
                  Crosshair
                </button>
                <button
                  onClick={() => setReticleStyle('corners')}
                  className={`px-2 py-1 rounded text-[11px] ${
                    reticleStyle === 'corners' ? 'bg-[#06b6d4] text-black font-semibold' : 'bg-[#1e293b] text-white'
                  }`}
                >
                  Corners
                </button>
              </div>
            </div>

            {/* Primary Action Controls */}
            <div className="flex flex-wrap items-center justify-between gap-3 pt-1 text-xs">
              <div className="flex items-center gap-2">
                {activeInputMode === 'camera' && (
                  <>
                    <select
                      value={selectedDeviceId}
                      onChange={(e) => {
                        setSelectedDeviceId(e.target.value);
                        if (isCameraActive) startCamera(e.target.value);
                      }}
                      className="bg-[#1e293b] border border-white/10 text-white rounded px-3 py-2 text-xs focus:outline-none focus:border-[#06b6d4]"
                    >
                      {devices.map((d, idx) => (
                        <option key={d.deviceId || idx} value={d.deviceId}>
                          {d.label || `Camera ${idx + 1}`}
                        </option>
                      ))}
                      {devices.length === 0 && <option value="">Default Webcam</option>}
                    </select>

                    {isCameraActive && (
                      <>
                        <button
                          onClick={togglePause}
                          className="px-3 py-2 bg-[#1e293b] border border-white/10 text-white rounded hover:bg-[#334155] transition-colors inline-flex items-center gap-1.5"
                        >
                          {isPaused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
                          <span>{isPaused ? 'Resume' : 'Pause'}</span>
                        </button>

                        <button
                          onClick={stopCamera}
                          className="px-3 py-2 bg-[#1e293b] border border-white/10 text-[#ef4444] rounded hover:bg-[#ef4444]/20 transition-colors inline-flex items-center gap-1.5"
                        >
                          <Square className="w-3.5 h-3.5" />
                          <span>Stop</span>
                        </button>
                      </>
                    )}
                  </>
                )}

                {activeInputMode !== 'camera' && (
                  <button
                    onClick={() => {
                      setActiveInputMode('camera');
                      startCamera(selectedDeviceId);
                    }}
                    className="px-4 py-2 bg-[#06b6d4] text-[#07090e] font-semibold rounded hover:bg-[#22d3ee] transition-all inline-flex items-center gap-1.5 shadow"
                  >
                    <Camera className="w-3.5 h-3.5" />
                    <span>Switch Back to Live Webcam</span>
                  </button>
                )}
              </div>

              <button
                onClick={handleLogItem}
                className="px-4 py-2 bg-[#10b981] text-[#07090e] font-semibold rounded hover:bg-[#059669] transition-colors inline-flex items-center gap-1.5 shadow"
              >
                <Save className="w-3.5 h-3.5" />
                <span>Log {selectedFruit.toUpperCase()} Snapshot</span>
              </button>
            </div>
          </div>

          {/* Right Column: Real-Time Diagnostic Telemetry */}
          <div className="lg:col-span-5 bg-[#0f141f] border border-white/10 rounded-xl p-6 space-y-6 shadow-2xl">
            {/* Variety Selector */}
            <div className="space-y-2 border-b border-white/10 pb-4">
              <div className="flex items-center justify-between text-xs text-[#94a3b8]">
                <span className="font-medium text-white">Target Fruit Classification</span>
                <span className="font-mono text-emerald-400 uppercase">{selectedFruit}</span>
              </div>
              <select
                value={selectedFruit}
                onChange={(e) => {
                  emaScoresRef.current = null;
                  setSelectedFruit(e.target.value);
                }}
                className="w-full bg-[#1e293b] border border-emerald-500/40 text-white rounded-lg px-4 py-2.5 text-xs font-medium focus:outline-none focus:border-emerald-400 cursor-pointer shadow-lg"
              >
                <option value="all">🌐 All Fruits (Auto-Detect)</option>
                <option value="mango">🥭 Mango (Alphonso, Kesar, Kent)</option>
                <option value="grapes">🍇 Grapes (Crimson, Thompson, Concord)</option>
                <option value="apple">🍎 Apples (Fuji, Gala, Granny Smith)</option>
                <option value="banana">🍌 Bananas (Cavendish, Robusta)</option>
                <option value="orange">🍊 Citrus (Valencia, Navel, Lemon)</option>
                <option value="strawberry">🍓 Strawberries & Berries</option>
                <option value="avocado">🥑 Avocados (Hass, Fuerte)</option>
              </select>
            </div>

            {/* AI Verdict Card */}
            <div className={`border rounded-xl p-6 text-center space-y-2 transition-all ${
              prediction === 'fresh' 
                ? 'bg-[#10b981]/10 border-[#10b981]/50 shadow-[0_0_25px_rgba(16,185,129,0.2)]'
                : prediction === 'rotten'
                ? 'bg-[#ef4444]/10 border-[#ef4444]/50 shadow-[0_0_25px_rgba(239,68,68,0.2)]'
                : 'bg-[#07090e] border-white/10'
            }`}>
              <span className="text-[10px] uppercase font-mono tracking-[0.2em] text-[#94a3b8] block">
                AI DIAGNOSTIC STATUS
              </span>

              {prediction === 'fresh' && (
                <div className="space-y-1">
                  <div className="text-4xl font-serif font-bold text-[#10b981] flex items-center justify-center gap-2">
                    <CheckCircle className="w-8 h-8" />
                    <span>FRESH</span>
                  </div>
                  <div className="text-xs font-mono text-[#10b981]/90">
                    High Cell Turgor • Uncompromised Epidermis ({(freshScore * 100).toFixed(1)}%)
                  </div>
                </div>
              )}

              {prediction === 'rotten' && (
                <div className="space-y-1">
                  <div className="text-4xl font-serif font-bold text-[#ef4444] flex items-center justify-center gap-2">
                    <AlertTriangle className="w-8 h-8" />
                    <span>ROTTEN</span>
                  </div>
                  <div className="text-xs font-mono text-[#ef4444]/90">
                    Pathological Decay • Necrotic Breakdown ({(rottenScore * 100).toFixed(1)}%)
                  </div>
                </div>
              )}

              {prediction === 'uncertain' && (
                <div className="space-y-1">
                  <div className="text-3xl font-serif font-bold text-[#f59e0b] flex items-center justify-center gap-2">
                    <HelpCircle className="w-7 h-7" />
                    <span>UNCERTAIN</span>
                  </div>
                  <div className="text-xs font-mono text-[#f59e0b]/90">
                    Calibrating Boundary • Adjust Distance or Lighting
                  </div>
                </div>
              )}

              {prediction === 'initializing' && (
                <div className="space-y-1">
                  <div className="text-2xl font-serif font-medium text-[#06b6d4] animate-pulse">
                    AWAITING SPECIMEN...
                  </div>
                  <div className="text-xs font-mono text-[#06b6d4]/80">
                    Position fruit inside center cyan reticle
                  </div>
                </div>
              )}

              {prediction === 'idle' && (
                <div className="text-2xl font-serif text-[#64748b]">
                  STANDBY
                </div>
              )}

              {/* Dynamic Fruit Guidance */}
              <p className="text-xs text-[#94a3b8] pt-2 italic leading-relaxed">
                {getFruitGuidance()}
              </p>
            </div>

            {/* Probability Meters */}
            <div className="space-y-4 text-xs font-sans">
              {/* Fresh Score */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-[#94a3b8]">
                  <span className="text-[#10b981] font-medium">Fresh Probability</span>
                  <span className="font-mono text-white">{(freshScore * 100).toFixed(1)}%</span>
                </div>
                <div className="h-2.5 bg-[#1e293b] rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[#10b981] transition-all duration-300 rounded-full"
                    style={{ width: `${Math.min(100, freshScore * 100)}%` }}
                  />
                </div>
              </div>

              {/* Rotten Score */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-[#94a3b8]">
                  <span className="text-[#ef4444] font-medium">Rotten Probability</span>
                  <span className="font-mono text-white">{(rottenScore * 100).toFixed(1)}%</span>
                </div>
                <div className="h-2.5 bg-[#1e293b] rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[#ef4444] transition-all duration-300 rounded-full"
                    style={{ width: `${Math.min(100, rottenScore * 100)}%` }}
                  />
                </div>
              </div>
            </div>

            {/* Spectral Pathology Telemetry */}
            <div className="bg-[#141b29] border border-white/10 rounded-xl p-4 space-y-3 shadow-inner">
              <div className="flex items-center justify-between text-[11px]">
                <span className="font-mono uppercase tracking-wider text-white/90 flex items-center gap-1.5">
                  <Activity className="w-3.5 h-3.5 text-emerald-400" />
                  <span>Spectral Pathology Telemetry</span>
                </span>
                <span className="text-[10px] font-mono text-cyan-400 bg-cyan-950/60 border border-cyan-800/40 px-2 py-0.5 rounded">
                  {engineType}
                </span>
              </div>

              {/* Biomarker Box */}
              <div className="bg-[#0a0e17] rounded-lg p-2.5 border border-white/5 space-y-1">
                <span className="text-[10px] font-mono uppercase text-[#64748b] block">Primary Biomarker Indication</span>
                <div className="text-xs font-mono font-medium text-white flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full flex-shrink-0 ${
                    prediction === 'fresh' ? 'bg-[#10b981]' : prediction === 'rotten' ? 'bg-[#ef4444]' : 'bg-[#f59e0b]'
                  }`} />
                  <span className="truncate">{detectedBiomarker}</span>
                </div>
              </div>

              {/* Dual Indices */}
              <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                <div className="bg-[#0a0e17] p-2.5 rounded-lg border border-white/5">
                  <div className="text-[10px] text-[#94a3b8] uppercase">Necrotic Area</div>
                  <div className={`text-base font-bold ${necroticPercent > 8 ? 'text-red-400' : 'text-emerald-400'}`}>
                    {necroticPercent.toFixed(1)}%
                  </div>
                  <div className="text-[9px] text-[#64748b]">Tissue Browning Index</div>
                </div>

                <div className="bg-[#0a0e17] p-2.5 rounded-lg border border-white/5">
                  <div className="text-[10px] text-[#94a3b8] uppercase">Healthy Pigment</div>
                  <div className={`text-base font-bold ${healthyPercent > 60 ? 'text-emerald-400' : 'text-amber-400'}`}>
                    {healthyPercent.toFixed(1)}%
                  </div>
                  <div className="text-[9px] text-[#64748b]">Carotenoid/Chlorophyll</div>
                </div>
              </div>
            </div>

            {/* Telemetry Metrics */}
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-[#1e293b]/70 border border-white/5 rounded-lg p-3.5 space-y-1">
                <span className="text-[10px] uppercase font-mono tracking-wider text-[#94a3b8] block">
                  INFERENCE LATENCY
                </span>
                <span className="font-mono text-lg text-white font-medium">
                  {inferenceMs !== null ? `${inferenceMs} ms` : '-- ms'}
                </span>
              </div>

              <div className="bg-[#1e293b]/70 border border-white/5 rounded-lg p-3.5 space-y-1">
                <span className="text-[10px] uppercase font-mono tracking-wider text-[#94a3b8] block">
                  LOGGED SPECIMENS
                </span>
                <span className="font-mono text-lg text-white font-medium">
                  {scanHistory.length}
                </span>
              </div>
            </div>

            {/* Threshold Slider */}
            <div className="space-y-2 pt-2 border-t border-white/10">
              <div className="flex justify-between text-xs text-[#94a3b8]">
                <span className="flex items-center gap-1.5">
                  <Sliders className="w-3.5 h-3.5 text-[#06b6d4]" />
                  <span>Certainty Decision Boundary</span>
                </span>
                <span className="font-mono text-[#06b6d4] font-semibold">{threshold}%</span>
              </div>
              <input
                type="range"
                min="50"
                max="95"
                step="5"
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
                className="w-full accent-[#06b6d4] bg-[#1e293b] h-1.5 rounded cursor-pointer"
              />
            </div>
          </div>
        </div>

        {/* LOGGED SNAPSHOT HISTORY FILMSTRIP (NEW) */}
        {scanHistory.length > 0 && (
          <div className="bg-[#0f141f] border border-white/10 rounded-xl p-5 space-y-3 shadow-xl">
            <div className="flex items-center justify-between text-xs text-[#94a3b8]">
              <span className="font-mono uppercase tracking-wider text-white flex items-center gap-2">
                <Save className="w-4 h-4 text-emerald-400" />
                <span>Recent Sample Test Snapshots (Filmstrip)</span>
              </span>
              <button
                onClick={() => setScanHistory([])}
                className="text-[11px] text-red-400 hover:text-red-300 inline-flex items-center gap-1"
              >
                <Trash2 className="w-3 h-3" />
                <span>Clear History</span>
              </button>
            </div>

            <div className="flex gap-4 overflow-x-auto pb-2 scrollbar-thin">
              {scanHistory.map((item) => (
                <div
                  key={item.id}
                  className="flex-shrink-0 w-36 bg-[#141b29] border border-white/5 rounded-lg p-2 space-y-1.5 text-xs"
                >
                  <div className="relative aspect-square w-full rounded overflow-hidden bg-black/50">
                    <img src={item.thumbUrl} alt={item.id} className="w-full h-full object-cover" />
                    <span
                      className={`absolute top-1 right-1 text-[9px] font-mono uppercase px-1 rounded font-bold ${
                        item.verdict === 'fresh'
                          ? 'bg-[#10b981] text-[#07090e]'
                          : item.verdict === 'rotten'
                          ? 'bg-[#ef4444] text-white'
                          : 'bg-[#f59e0b] text-black'
                      }`}
                    >
                      {item.verdict}
                    </span>
                  </div>
                  <div className="font-medium text-white truncate">{item.fruit}</div>
                  <div className="text-[10px] text-[#94a3b8] flex justify-between font-mono">
                    <span>{item.timestamp}</span>
                    <span>{(item.confidence * 100).toFixed(0)}%</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </section>
  );
};
