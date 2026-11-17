import React, { useState } from 'react';
import { Navigation } from './components/Navigation';
import { Hero } from './components/Hero';
import { LiveCameraScanner } from './components/LiveCameraScanner';
import { EditorialIntro } from './components/EditorialIntro';
import { Showcase } from './components/Showcase';
import { Philosophy } from './components/Philosophy';
import { FinalCTA } from './components/FinalCTA';
import { Footer } from './components/Footer';
import { WorkDetailModal } from './components/WorkDetailModal';
import { InquiryModal } from './components/InquiryModal';
import { Artwork } from './data/content';
import { useScrollReveal } from './hooks/useScrollReveal';

export const App: React.FC = () => {
  const [selectedArtwork, setSelectedArtwork] = useState<Artwork | null>(null);
  const [inquiryOpen, setInquiryOpen] = useState(false);
  const [inquirySubject, setInquirySubject] = useState<string>('Agricultural Quality Inspection');

  useScrollReveal();

  const handleOpenInquiry = (subjectTitle?: string) => {
    if (subjectTitle) setInquirySubject(subjectTitle);
    setInquiryOpen(true);
  };

  const scrollToWorks = () => {
    const el = document.getElementById('works');
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const scrollToScanner = () => {
    const el = document.getElementById('scanner');
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className="min-h-screen bg-[#fafaf9] text-[#0f172a] flex flex-col selection:bg-[#0f172a] selection:text-white">
      {/* A. Minimal Navigation (Matches Image 1) */}
      <Navigation
        onOpenInquiry={() => handleOpenInquiry('Laboratory & Pilot Deployment Inquiry')}
        onOpenScanner={scrollToScanner}
      />

      <main className="flex-1">
        {/* B. Full-Screen Cinematic Hero (Matches Image 1) */}
        <Hero
          onExploreClick={scrollToWorks}
          onOpenInquiry={() => handleOpenInquiry('Agricultural Facility Inquiry')}
          onOpenScanner={scrollToScanner}
        />

        {/* C. Live Interactive Camera Scanner (Matches Image 2) */}
        <LiveCameraScanner />

        {/* D. Editorial Introduction */}
        <EditorialIntro />

        {/* E. Featured Work / Quality Studies */}
        <Showcase onSelectArtwork={(art) => setSelectedArtwork(art)} />

        {/* F. Neural Philosophy & Verification Audits */}
        <Philosophy />

        {/* G. Final Call to Action */}
        <FinalCTA onOpenInquiry={() => handleOpenInquiry('Custom Facility Quality Pilot')} />
      </main>

      {/* H. Minimal Footer */}
      <Footer />

      {/* Interactive Curatorial Modals */}
      <WorkDetailModal
        artwork={selectedArtwork}
        onClose={() => setSelectedArtwork(null)}
        onOpenInquiry={handleOpenInquiry}
      />

      <InquiryModal
        isOpen={inquiryOpen}
        onClose={() => setInquiryOpen(false)}
        defaultSubject={inquirySubject}
      />
    </div>
  );
};

export default App;
