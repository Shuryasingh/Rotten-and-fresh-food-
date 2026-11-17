import React from 'react';
import { Camera, ArrowUpRight } from 'lucide-react';

interface FinalCTAProps {
  onOpenInquiry: () => void;
}

export const FinalCTA: React.FC<FinalCTAProps> = ({ onOpenInquiry }) => {
  return (
    <section className="relative bg-[#ffffff] text-[#0f172a] py-32 md:py-44 px-6 md:px-12 lg:px-16 border-t border-[rgba(15,23,42,0.06)] overflow-hidden">
      <div className="max-w-[1440px] mx-auto text-center space-y-12">
        {/* Curatorial Header */}
        <div className="reveal-on-scroll">
          <span className="editorial-tag text-[hsl(215,25%,32%)] block">
            Section IV / Deploy & Inspect
          </span>
        </div>

        {/* Large Bold Minimal Serif Closing Statement */}
        <div className="reveal-on-scroll max-w-4xl mx-auto space-y-6">
          <h2 className="font-serif text-4xl sm:text-6xl md:text-7xl lg:text-[76px] leading-[1.0] tracking-tight text-[#0f172a]">
            Deploy real-time fruit freshness intelligence across your pipeline.
          </h2>

          <p className="text-[hsl(215,25%,32%)] font-sans text-base md:text-lg max-w-xl mx-auto leading-relaxed">
            Ready for live evaluation on integrated laptop webcams, industrial USB optics, or standalone Raspberry Pi 5 edge rigs.
          </p>
        </div>

        {/* High-Contrast Black CTA Button */}
        <div className="reveal-on-scroll pt-4 flex flex-col sm:flex-row items-center justify-center gap-4">
          <a
            href="http://localhost:8080"
            target="_blank"
            rel="noopener noreferrer"
            className="premium-button w-full sm:w-auto px-10 py-5 bg-[#000000] text-[#ffffff] text-[14px] font-medium tracking-[0.16em] uppercase border border-black hover:bg-[#1a1a1a] transition-all duration-300 inline-flex items-center justify-center gap-3"
          >
            <Camera className="w-4 h-4 text-emerald-400" />
            <span>Launch Live Camera Scanner</span>
          </a>

          <a
            href="#works"
            className="premium-button w-full sm:w-auto px-8 py-5 bg-transparent text-[#0f172a] text-[14px] font-medium tracking-[0.16em] uppercase border border-[rgba(15,23,42,0.2)] hover:border-[#0f172a] transition-all duration-300 inline-flex items-center justify-center gap-2"
          >
            <span>Review Specimen Studies</span>
            <ArrowUpRight className="w-3.5 h-3.5" />
          </a>

          <button
            onClick={onOpenInquiry}
            className="premium-button w-full sm:w-auto px-8 py-5 bg-[#fafaf9] text-[#0f172a] text-[14px] font-medium tracking-[0.16em] uppercase border border-[rgba(15,23,42,0.15)] hover:border-black transition-all duration-300 inline-flex items-center justify-center gap-2"
          >
            <span>Custom Quality Audits</span>
            <ArrowUpRight className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Locations Footnote */}
        <div className="reveal-on-scroll pt-16 flex flex-wrap justify-center gap-8 md:gap-14 text-xs font-sans text-[hsl(215,25%,32%)] tracking-[0.2em] uppercase">
          <span>Paris Quality Lab</span>
          <span>•</span>
          <span>Kyoto Agricultural Guild</span>
          <span>•</span>
          <span>Basel Cold-Chain Nexus</span>
        </div>
      </div>
    </section>
  );
};
