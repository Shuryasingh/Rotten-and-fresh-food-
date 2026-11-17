import React from 'react';
import { SITE_METADATA } from '../data/content';

export const EditorialIntro: React.FC = () => {
  return (
    <section
      id="editorial-intro"
      className="relative bg-[#fafaf9] text-[#0f172a] py-24 md:py-36 px-6 md:px-12 lg:px-16 border-b border-[rgba(15,23,42,0.06)] overflow-hidden"
    >
      <div className="max-w-[1440px] mx-auto">
        {/* Top Editorial Index Line */}
        <div className="reveal-on-scroll flex flex-col sm:flex-row sm:items-center justify-between gap-4 text-xs tracking-[0.2em] uppercase text-[hsl(215,25%,32%)] font-sans border-b border-[rgba(15,23,42,0.08)] pb-6 mb-16 md:mb-24">
          <div className="flex items-center gap-3">
            <span className="font-semibold text-[#0f172a]">SECTION I</span>
            <span>/</span>
            <span>BOTANICAL QUALITY MANIFESTO</span>
          </div>
          <div>{SITE_METADATA.edition}</div>
        </div>

        {/* Asymmetrical Editorial Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-16 items-start">
          {/* Left Column: Oversized Serif Headline */}
          <div className="lg:col-span-7 space-y-6">
            <span className="editorial-tag text-[hsl(215,25%,32%)] block">
              The Geometry of Freshness
            </span>
            <h2 className="reveal-on-scroll font-serif text-4xl sm:text-5xl md:text-6xl lg:text-[68px] leading-[1.05] tracking-tight text-[#0f172a]">
              We evaluate visible food condition with mathematical restraint and zero guesswork.
            </h2>
          </div>

          {/* Right Column: Minimal Narrative & Storytelling */}
          <div className="lg:col-span-5 space-y-8 lg:pt-6">
            <p className="reveal-on-scroll text-[hsl(215,25%,32%)] font-sans text-base md:text-[18px] leading-relaxed font-normal">
              {SITE_METADATA.brandName} merges deep convolutional neural networks with high-end editorial clarity. By scanning the epidermal cuticle of apples, bananas, and citrus under calibrated lighting, our vision pipeline distinguishes benign cosmetic ripening from active bacterial and fungal breakdown.
            </p>

            <p className="reveal-on-scroll text-[hsl(215,25%,32%)] font-sans text-sm md:text-base leading-relaxed text-[hsl(215,25%,32%)]">
              Equipped with a strict 3-state output (<span className="text-emerald-700 font-medium">FRESH</span>, <span className="text-red-700 font-medium">ROTTEN</span>, or <span className="text-amber-700 font-medium">UNCERTAIN</span>), our system never gambles with operator certainty. Every frame is processed in 12.4ms with zero cloud dependency.
            </p>

            {/* Editorial Metadata Box */}
            <div className="reveal-on-scroll pt-6 border-t border-[rgba(15,23,42,0.08)] grid grid-cols-2 gap-6 text-xs">
              <div>
                <span className="block text-[hsl(215,25%,32%)] uppercase tracking-[0.16em] mb-1 font-medium">
                  Architecture
                </span>
                <span className="text-[#0f172a] font-serif text-lg">
                  MobileNetV2 • ONNX
                </span>
              </div>
              <div>
                <span className="block text-[hsl(215,25%,32%)] uppercase tracking-[0.16em] mb-1 font-medium">
                  Rotten Recall
                </span>
                <span className="text-[#0f172a] font-serif text-lg">
                  99.1% Spoilage Caught
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
