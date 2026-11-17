import React, { useEffect } from 'react';
import { Artwork } from '../data/content';
import { X, ArrowUpRight, MapPin } from 'lucide-react';

interface WorkDetailModalProps {
  artwork: Artwork | null;
  onClose: () => void;
  onOpenInquiry: (subjectTitle?: string) => void;
}

export const WorkDetailModal: React.FC<WorkDetailModalProps> = ({
  artwork,
  onClose,
  onOpenInquiry,
}) => {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    if (artwork) {
      document.body.style.overflow = 'hidden';
      window.addEventListener('keydown', handleKeyDown);
    }
    return () => {
      document.body.style.overflow = '';
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [artwork, onClose]);

  if (!artwork) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 md:p-10 bg-[#0f172a]/85 backdrop-blur-md animate-fade-rise">
      <div
        className="relative bg-[#ffffff] w-full max-w-5xl max-h-[92vh] overflow-y-auto text-[#0f172a] shadow-2xl border border-[rgba(15,23,42,0.1)] flex flex-col md:flex-row"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close Button */}
        <button
          onClick={onClose}
          aria-label="Close dialog"
          className="absolute top-4 right-4 z-20 p-2 bg-white/80 hover:bg-black hover:text-white transition-colors border border-black/10"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Left Column: Image */}
        <div className="md:w-1/2 bg-[#f5f5f4] relative min-h-[360px] md:min-h-full">
          <img
            src={artwork.imageUrl}
            alt={artwork.title}
            loading="lazy"
            decoding="async"
            className="w-full h-full object-cover"
          />
          <div className="absolute bottom-4 left-4">
            <span className="bg-black/80 text-white text-[10px] font-sans uppercase tracking-[0.2em] px-3 py-1">
              Cat. {artwork.id}
            </span>
          </div>
        </div>

        {/* Right Column: Curatorial Details */}
        <div className="md:w-1/2 p-8 sm:p-10 md:p-12 flex flex-col justify-between space-y-8">
          <div className="space-y-6">
            <div className="space-y-2">
              <span className="editorial-tag text-[hsl(215,25%,32%)] block">
                {artwork.category} • {artwork.year}
              </span>
              <h3 className="font-serif text-3xl sm:text-4xl md:text-5xl text-[#0f172a] leading-tight">
                {artwork.title}
              </h3>
              <p className="text-sm font-sans text-[hsl(215,25%,32%)]">
                {artwork.subtitle}
              </p>
            </div>

            {/* Technical Specifications */}
            <div className="border-t border-b border-[rgba(15,23,42,0.08)] py-6 space-y-3 text-xs font-sans">
              <div className="flex justify-between">
                <span className="text-[hsl(215,25%,32%)] uppercase tracking-wider">Medium:</span>
                <span className="font-medium text-[#0f172a] text-right">{artwork.medium}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[hsl(215,25%,32%)] uppercase tracking-wider">Dimensions:</span>
                <span className="font-medium text-[#0f172a]">{artwork.dimensions}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-[hsl(215,25%,32%)] uppercase tracking-wider">Provenance:</span>
                <span className="font-medium text-[#0f172a] inline-flex items-center gap-1">
                  <MapPin className="w-3 h-3 text-[hsl(215,25%,32%)]" />
                  {artwork.location}
                </span>
              </div>
            </div>

            {/* Curatorial Notes */}
            <div className="space-y-2">
              <span className="editorial-tag text-[hsl(215,25%,32%)] block">
                Curator’s Essay Note
              </span>
              <p className="font-sans text-sm text-[hsl(215,25%,32%)] leading-relaxed italic">
                “{artwork.curatorNotes}”
              </p>
            </div>
          </div>

          {/* Action Row */}
          <div className="pt-6 border-t border-[rgba(15,23,42,0.08)] flex items-center justify-between gap-4">
            <button
              onClick={() => {
                onClose();
                onOpenInquiry(`Acquisition Inquiry: ${artwork.title}`);
              }}
              className="premium-button flex-1 px-6 py-4 bg-[#000000] text-[#ffffff] text-xs font-sans font-medium uppercase tracking-[0.16em] inline-flex items-center justify-center gap-2"
            >
              <span>Request Acquisition Dossier</span>
              <ArrowUpRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
