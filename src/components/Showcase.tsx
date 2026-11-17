import React, { useState } from 'react';
import { FEATURED_ARTWORKS, Artwork } from '../data/content';
import { ArrowUpRight, Eye } from 'lucide-react';

interface ShowcaseProps {
  onSelectArtwork: (artwork: Artwork) => void;
}

export const Showcase: React.FC<ShowcaseProps> = ({ onSelectArtwork }) => {
  const [activeCategory, setActiveCategory] = useState<string>('All');

  const categories = [
    'All',
    'Apples',
    'Bananas',
    'Citrus',
    'Mango',
    'Grapes',
    'Exotic & Berries',
  ];

  const filteredArtworks =
    activeCategory === 'All'
      ? FEATURED_ARTWORKS
      : FEATURED_ARTWORKS.filter((item) => item.category === activeCategory);

  return (
    <section id="works" className="relative bg-[#ffffff] text-[#0f172a] py-28 md:py-40 px-6 md:px-12 lg:px-16">
      <div className="max-w-[1440px] mx-auto">
        {/* Section Header */}
        <div className="reveal-on-scroll flex flex-col md:flex-row md:items-end justify-between gap-8 border-b border-[rgba(15,23,42,0.08)] pb-10 mb-16">
          <div className="space-y-3">
            <span className="editorial-tag text-[hsl(215,25%,32%)] block">
              Section II / Multi-Fruit Specimen Archive
            </span>
            <h2 className="font-serif text-4xl sm:text-5xl md:text-6xl text-[#0f172a] tracking-tight">
              Anatomy of Produce & Spoilage
            </h2>
            <p className="text-xs sm:text-sm text-[hsl(215,25%,32%)] font-sans">
              Apples, Bananas, Citrus, Mangoes, Grapes, Strawberries & Avocados under macro spectral vision.
            </p>
          </div>

          {/* Minimal Fruit Category Tabs */}
          <div className="flex flex-wrap items-center gap-2 md:gap-4 text-xs font-sans tracking-widest uppercase">
            {categories.map((cat) => (
              <button
                key={cat}
                onClick={() => setActiveCategory(cat)}
                className={`py-1.5 px-3 transition-all duration-300 ${
                  activeCategory === cat
                    ? 'text-[#0f172a] font-semibold border-b-2 border-[#000000]'
                    : 'text-[hsl(215,25%,32%)] hover:text-[#0f172a]'
                }`}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>

        {/* Asymmetrical Showcase Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-10 md:gap-14 lg:gap-16">
          {filteredArtworks.map((artwork, index) => {
            const isLarge = index === 0 || index === 4;

            return (
              <div
                key={artwork.id}
                onClick={() => onSelectArtwork(artwork)}
                className={`group cursor-pointer reveal-on-scroll flex flex-col ${
                  isLarge ? 'md:col-span-2' : 'col-span-1'
                }`}
              >
                {/* Media Frame */}
                <div className="relative overflow-hidden bg-[#f5f5f4] aspect-[4/5] sm:aspect-[3/4] md:aspect-auto md:h-[520px] lg:h-[580px] w-full">
                  <img
                    src={artwork.imageUrl}
                    alt={artwork.title}
                    loading="lazy"
                    className="w-full h-full object-cover transition-transform duration-700 ease-out group-hover:scale-105"
                  />

                  {/* Hover Overlay */}
                  <div className="absolute inset-0 bg-[#0f172a]/25 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center pointer-events-none">
                    <span className="inline-flex items-center gap-2 bg-[#ffffff] text-[#0f172a] px-5 py-2.5 text-xs font-sans uppercase tracking-[0.18em] shadow-lg">
                      <Eye className="w-3.5 h-3.5" />
                      <span>Examine Fruit Case Study</span>
                    </span>
                  </div>

                  {/* Category Pill */}
                  <div className="absolute top-5 right-5 z-10">
                    <span className="bg-white/95 backdrop-blur-md text-[#0f172a] text-[10px] font-sans font-medium uppercase tracking-[0.2em] px-3 py-1 shadow-sm">
                      {artwork.category}
                    </span>
                  </div>
                </div>

                {/* Typography & Metadata */}
                <div className="pt-6 space-y-2 border-b border-[rgba(15,23,42,0.06)] pb-6 flex-1 flex flex-col justify-between">
                  <div className="space-y-1">
                    <div className="flex items-baseline justify-between gap-4">
                      <h3 className="font-serif text-2xl md:text-3xl text-[#0f172a] group-hover:text-black transition-colors">
                        {artwork.title}
                      </h3>
                      <ArrowUpRight className="w-4 h-4 text-[#0f172a] opacity-0 group-hover:opacity-100 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all duration-300" />
                    </div>
                    <p className="text-[hsl(215,25%,32%)] font-sans text-xs md:text-sm tracking-normal">
                      {artwork.subtitle}
                    </p>
                  </div>

                  <div className="pt-4 flex items-center justify-between text-[11px] font-sans text-[hsl(215,25%,32%)] uppercase tracking-[0.15em]">
                    <span>{artwork.location}</span>
                    <span className="opacity-70">{artwork.dimensions}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
};
