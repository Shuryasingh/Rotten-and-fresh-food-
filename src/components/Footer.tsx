import React from 'react';
import { ArrowUp } from 'lucide-react';
import { SITE_METADATA } from '../data/content';

export const Footer: React.FC = () => {
  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <footer className="bg-[#0f172a] text-white pt-24 pb-16 px-6 md:px-12 lg:px-16 border-t border-white/10">
      <div className="max-w-[1440px] mx-auto space-y-20">
        {/* Top Footer: Editorial Wordmark & Scroll to Top */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-10 border-b border-white/10 pb-16">
          <div className="space-y-4">
            <span className="text-[11px] font-sans uppercase tracking-[0.25em] text-white/50 block">
              {SITE_METADATA.edition}
            </span>
            <h2 className="font-serif text-4xl sm:text-5xl md:text-6xl lg:text-7xl tracking-tight leading-none text-white">
              {SITE_METADATA.brandName}
            </h2>
          </div>

          <button
            onClick={scrollToTop}
            aria-label="Return to zenith of page"
            className="group self-start md:self-auto inline-flex items-center gap-3 text-xs uppercase tracking-[0.2em] font-sans text-white/70 hover:text-white transition-colors"
          >
            <span>Zenith</span>
            <div className="w-9 h-9 rounded-full border border-white/20 flex items-center justify-center group-hover:border-white transition-colors">
              <ArrowUp className="w-3.5 h-3.5 group-hover:-translate-y-0.5 transition-transform" />
            </div>
          </button>
        </div>

        {/* Middle Footer: Editorial Columns */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-12 text-sm">
          {/* Column 1: Locations */}
          <div className="space-y-4">
            <span className="text-xs uppercase tracking-[0.2em] text-white/50 block font-medium">
              Research Hubs
            </span>
            <ul className="space-y-2 text-white/80 font-sans text-xs leading-relaxed">
              <li>14 Rue de l’Université, 75007 Paris</li>
              <li>42 Gion-machi, Higashiyama, Kyoto</li>
              <li>Münsterplatz 16, 4051 Basel</li>
              <li className="pt-2 text-white/40 italic">Edge deployment: Raspberry Pi 5 & PC</li>
            </ul>
          </div>

          {/* Column 2: Essential Navigation */}
          <div className="space-y-4">
            <span className="text-xs uppercase tracking-[0.2em] text-white/50 block font-medium">
              Inspection Index
            </span>
            <ul className="space-y-2 text-white/80 font-sans text-xs">
              <li>
                <a href="#works" className="hover:text-white transition-colors">
                  Specimen Archive & Studies
                </a>
              </li>
              <li>
                <a href="#philosophy" className="hover:text-white transition-colors">
                  Philosophy of Rotten Recall
                </a>
              </li>
              <li>
                <a href="#audits" className="hover:text-white transition-colors">
                  Zero Data-Leakage Audit
                </a>
              </li>
              <li>
                <a href="http://localhost:8080" target="_blank" rel="noopener noreferrer" className="hover:text-emerald-400 transition-colors">
                  Live Camera Scanner (Port 8080)
                </a>
              </li>
            </ul>
          </div>

          {/* Column 3: Standards */}
          <div className="space-y-4">
            <span className="text-xs uppercase tracking-[0.2em] text-white/50 block font-medium">
              Verification Standards
            </span>
            <ul className="space-y-2 text-white/80 font-sans text-xs">
              <li>
                <span className="text-white/60">Strict 3-State Verdict:</span> Fresh / Rotten / Uncertain
              </li>
              <li>
                <span className="text-white/60">Threshold:</span> 80% Softmax Confidence
              </li>
              <li>
                <span className="text-white/60">Model Format:</span> MobileNetV2 ONNX
              </li>
              <li>
                <span className="text-white/60">Logging:</span> logs/scans.csv Real-time
              </li>
            </ul>
          </div>

          {/* Column 4: Channels */}
          <div className="space-y-4">
            <span className="text-xs uppercase tracking-[0.2em] text-white/50 block font-medium">
              Engineering Links
            </span>
            <ul className="space-y-2 text-white/80 font-sans text-xs">
              <li>
                <a href="http://localhost:8080" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">
                  Launch Web GUI (Browser Camera)
                </a>
              </li>
              <li>
                <span className="text-white/60">OpenCV CLI:</span> launch.bat
              </li>
              <li>
                <span className="text-white/60">Dataset:</span> Apples, Bananas, Oranges
              </li>
              <li>
                <span className="text-white/60">Status:</span> 100% Offline Capable
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom Footer: Legal */}
        <div className="pt-12 border-t border-white/10 flex flex-col sm:flex-row items-center justify-between gap-6 text-xs text-white/50 font-sans">
          <div>
            © {new Date().getFullYear()} Rotten and Fresh Food Checker. MobileNetV2 Neural Inspection Platform.
          </div>
          <div className="flex items-center gap-6">
            <span>Instrument Serif & Inter</span>
            <span>•</span>
            <span>Minimalist Editorial Vision</span>
          </div>
        </div>
      </div>
    </footer>
  );
};
