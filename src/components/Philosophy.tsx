import React from 'react';
import { PHILOSOPHY_PILLARS, VERIFICATION_AUDITS } from '../data/content';
import { CheckCircle2, ArrowUpRight, Camera } from 'lucide-react';

export const Philosophy: React.FC = () => {
  return (
    <section id="philosophy" className="relative bg-[#fafaf9] text-[#0f172a] py-28 md:py-40 px-6 md:px-12 lg:px-16 border-t border-[rgba(15,23,42,0.06)]">
      <div className="max-w-[1440px] mx-auto space-y-28 md:space-y-36">
        {/* Editorial Heading Statement */}
        <div className="max-w-4xl space-y-8">
          <div className="reveal-on-scroll">
            <span className="editorial-tag text-[hsl(215,25%,32%)] block">
              Section III / Neural Integrity & Discipline
            </span>
          </div>

          <h2 className="reveal-on-scroll font-serif text-4xl sm:text-5xl md:text-6xl lg:text-[70px] leading-[1.04] text-[#0f172a] tracking-tight">
            “In food safety, a missed rotten fruit spoils the entire harvest.”
          </h2>

          <p className="reveal-on-scroll text-[hsl(215,25%,32%)] font-sans text-base md:text-lg max-w-2xl leading-relaxed">
            Conventional classification systems chase misleading overall accuracy percentages by sacrificing recall. We architected MobileNetV2 with an asymmetrical penalty function, guaranteeing that deteriorating produce is isolated before ethylene gas triggers chain-reaction spoilage.
          </p>
        </div>

        {/* 3 Core Pillars: Asymmetric Editorial Columns */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-12 lg:gap-16 pt-8 border-t border-[rgba(15,23,42,0.08)]">
          {PHILOSOPHY_PILLARS.map((pillar) => (
            <div key={pillar.number} className="reveal-on-scroll space-y-5">
              <span className="font-serif text-3xl md:text-4xl text-[#0f172a]/30 block">
                {pillar.number}
              </span>
              <h3 className="font-serif text-2xl md:text-3xl text-[#0f172a] tracking-tight">
                {pillar.title}
              </h3>
              <p className="font-sans font-medium text-xs uppercase tracking-[0.16em] text-[#0f172a]">
                {pillar.statement}
              </p>
              <p className="font-sans text-sm text-[hsl(215,25%,32%)] leading-relaxed">
                {pillar.description}
              </p>
            </div>
          ))}
        </div>

        {/* Verification Audits Table */}
        <div id="audits" className="pt-16 border-t border-[rgba(15,23,42,0.08)]">
          <div className="reveal-on-scroll flex flex-col sm:flex-row sm:items-end justify-between gap-4 mb-12">
            <div>
              <span className="editorial-tag text-[hsl(215,25%,32%)] block mb-2">
                Technical Benchmark & Parity Audits
              </span>
              <h3 className="font-serif text-3xl md:text-4xl text-[#0f172a]">
                Mathematical Parity Verification
              </h3>
            </div>
            <a
              href="http://localhost:8080"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs font-sans uppercase tracking-[0.2em] text-[#0f172a] inline-flex items-center gap-1.5 hover:underline"
            >
              <span>Test Live Inference</span>
              <Camera className="w-3.5 h-3.5" />
            </a>
          </div>

          <div className="divide-y divide-[rgba(15,23,42,0.08)]">
            {VERIFICATION_AUDITS.map((audit) => (
              <div
                key={audit.code}
                className="reveal-on-scroll group py-8 grid grid-cols-1 md:grid-cols-12 gap-4 md:gap-8 items-baseline hover:bg-white/70 transition-colors px-4 -mx-4"
              >
                <div className="md:col-span-2 text-xs font-sans tracking-widest text-[hsl(215,25%,32%)] uppercase">
                  {audit.code}
                </div>

                <div className="md:col-span-5 font-serif text-2xl md:text-3xl text-[#0f172a] group-hover:translate-x-1 transition-transform">
                  {audit.title}
                </div>

                <div className="md:col-span-3 text-sm font-sans text-[hsl(215,25%,32%)]">
                  <div>{audit.category}</div>
                  <div className="text-xs opacity-75 font-mono">{audit.metric}</div>
                </div>

                <div className="md:col-span-2 text-right flex items-center justify-between md:justify-end gap-3">
                  <span className="inline-flex items-center gap-1.5 text-xs font-sans font-medium tracking-wider text-emerald-800 bg-emerald-50 px-2.5 py-1 border border-emerald-200">
                    <CheckCircle2 className="w-3 h-3 text-emerald-600" />
                    <span>{audit.benchmark}</span>
                  </span>
                  <ArrowUpRight className="w-4 h-4 text-[#0f172a] opacity-0 group-hover:opacity-100 transition-opacity hidden md:inline-block" />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};
