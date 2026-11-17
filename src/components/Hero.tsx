import React from 'react';
import { ArrowDown, Scan, Camera } from 'lucide-react';

interface HeroProps {
  onExploreClick: () => void;
  onOpenInquiry: () => void;
  onOpenScanner?: () => void;
}

export const Hero: React.FC<HeroProps> = ({ onExploreClick, onOpenInquiry, onOpenScanner }) => {
  const handleLaunchScanner = () => {
    if (onOpenScanner) {
      onOpenScanner();
    } else {
      const el = document.getElementById('scanner');
      if (el) el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <section className="hero relative min-h-screen w-full flex flex-col justify-between px-6 md:px-12 lg:px-16 pt-32 pb-10 select-none">
      {/* Background Produce Imagery (Optimized for faster initial load) */}
      <div className="absolute inset-0 -z-2 overflow-hidden">
        <img
          src="https://images.unsplash.com/photo-1542838132-92c53300491e?q=80&w=1200&auto=format&fit=crop"
          alt="Fresh Produce Quality Display"
          loading="eager"
          fetchPriority="high"
          decoding="async"
          className="w-full h-full object-cover object-center filter brightness-[0.72] contrast-[1.08]"
        />
      </div>

      {/* Darkened Vignette Overlay for Crisp Typography Legibility */}
      <div className="absolute inset-0 -z-1 bg-gradient-to-b from-black/65 via-black/40 to-black/85 pointer-events-none" />

      {/* Top Editorial Status Bar (Matches Image 1) */}
      <div className="w-full max-w-[1440px] mx-auto flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-white/90 text-[11px] font-sans tracking-[0.2em] uppercase border-b border-white/20 pb-4 fade-rise-0">
        <div className="flex items-center gap-2.5">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="font-medium text-emerald-300">ONNX MOBILENETV2 ACTIVE</span>
          <span className="text-white/50">•</span>
          <span className="text-white/80">PRECISION INSPECTION ENGINE</span>
        </div>
        <div className="flex items-center gap-6 text-white/80">
          <span>RECALL: <strong className="text-white font-semibold">99.1% ROTTEN</strong></span>
          <span className="text-white/40">|</span>
          <span>LATENCY: <strong className="text-white font-semibold">12.4MS</strong></span>
          <span className="text-white/40">|</span>
          <span>STATUS: <strong className="text-emerald-400 font-semibold">100% OFFLINE READY</strong></span>
        </div>
      </div>

      {/* Main Center-Left Editorial Composition (Matches Image 1) */}
      <div className="w-full max-w-[1440px] mx-auto my-auto py-10 md:py-16">
        <div className="max-w-4xl space-y-7">
          {/* Subtitle Tagline */}
          <div className="fade-rise-0">
            <span className="inline-block text-white/90 font-sans text-xs tracking-[0.28em] uppercase border-l-2 border-emerald-400 pl-3">
              NEURAL VISION & FRESHNESS INTELLIGENCE
            </span>
          </div>

          {/* Mandatory H1 Headline — Exactly -2.46px letter-spacing, weight 400 */}
          <h1
            className="hero-title text-white text-balance drop-shadow-[0_4px_24px_rgba(0,0,0,0.85)] fade-rise-0"
            style={{
              fontFamily: '"Instrument Serif", serif',
              fontWeight: 400,
              lineHeight: 0.95,
              letterSpacing: '-2.46px',
            }}
          >
            Rotten and Fresh Food Checker
          </h1>

          {/* Subheading: 18px, Inter, weight 400 */}
          <p className="fade-rise-1 text-white/95 font-sans text-[18px] md:text-[20px] font-normal leading-relaxed max-w-2xl tracking-normal drop-shadow-[0_2px_14px_rgba(0,0,0,0.8)]">
            A high-fashion editorial lens on agricultural quality control. Classifying botanical produce into Fresh, Rotten, or Uncertain with mathematical rigor and zero-latency deep learning.
          </p>

          {/* 3 Action Buttons (Exact styling from Image 1) */}
          <div className="fade-rise-2 flex flex-wrap items-center gap-4 md:gap-5 pt-3">
            {/* 1. Launch Live Camera Scanner */}
            <button
              onClick={handleLaunchScanner}
              className="premium-button px-7 py-4 bg-[#0a0f1d]/90 text-white text-[13px] font-medium tracking-wider uppercase border border-white/30 hover:border-emerald-400 hover:bg-black transition-all duration-300 inline-flex items-center gap-2.5 shadow-xl"
            >
              <Camera className="w-4 h-4 text-emerald-400" />
              <span>LAUNCH LIVE CAMERA SCANNER</span>
            </button>

            {/* 2. Inspect Specimen Archive */}
            <button
              onClick={onExploreClick}
              className="premium-button px-7 py-4 bg-white/10 backdrop-blur-md text-white text-[13px] font-medium tracking-wider uppercase border border-white/25 hover:bg-white/25 hover:border-white transition-all duration-300 inline-flex items-center gap-2.5 shadow-xl"
            >
              <Scan className="w-4 h-4 text-white/90" />
              <span>INSPECT SPECIMEN ARCHIVE</span>
            </button>

            {/* 3. Lab Inquiries */}
            <button
              onClick={onOpenInquiry}
              className="premium-button px-7 py-4 bg-white/10 backdrop-blur-md text-white text-[13px] font-medium tracking-wider uppercase border border-white/25 hover:bg-white/25 hover:border-white transition-all duration-300 shadow-xl"
            >
              LAB INQUIRIES
            </button>
          </div>
        </div>
      </div>

      {/* Hero Bottom Bar with Scroll Cue */}
      <div className="w-full max-w-[1440px] mx-auto flex items-end justify-between text-white/80 pt-6 border-t border-white/20 fade-rise-3">
        <div className="text-[12px] font-sans tracking-wide max-w-sm hidden lg:block text-white/70">
          MobileNetV2 CNN calibrated with strict physical specimen isolation. Zero training leakage across splits.
        </div>

        <button
          onClick={onExploreClick}
          aria-label="Scroll to explore quality studies"
          className="group inline-flex items-center gap-3 text-xs tracking-[0.2em] uppercase font-sans text-white hover:text-emerald-300 transition-colors"
        >
          <span>EXPLORE QUALITY STUDIES</span>
          <ArrowDown className="w-4 h-4 transition-transform duration-300 group-hover:translate-y-1" />
        </button>

        <div className="hidden sm:block text-[11px] font-sans uppercase tracking-[0.18em] text-white/70">
          LIVE INSPECTION READY
        </div>
      </div>
    </section>
  );
};
