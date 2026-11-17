import React, { useState, useEffect } from 'react';
import { Menu, X, Camera } from 'lucide-react';

interface NavigationProps {
  onOpenInquiry: () => void;
  onOpenScanner?: () => void;
}

export const Navigation: React.FC<NavigationProps> = ({ onOpenInquiry, onOpenScanner }) => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 40) {
        setIsScrolled(true);
      } else {
        setIsScrolled(false);
      }
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const navLinks = [
    { label: 'Quality Studies', href: '#works' },
    { label: 'Neural Philosophy', href: '#philosophy' },
    { label: 'Verification Audits', href: '#audits' },
    { label: 'Live Scanner', href: '#scanner' },
  ];

  const handleCameraScanClick = () => {
    if (onOpenScanner) {
      onOpenScanner();
    } else {
      const el = document.getElementById('scanner');
      if (el) el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <>
      <header
        className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ease-out ${
          isScrolled
            ? 'bg-[#0f141f]/95 backdrop-blur-md border-b border-white/10 py-4 shadow-xl'
            : 'bg-transparent py-5 md:py-7'
        }`}
      >
        <div className="max-w-[1440px] mx-auto px-6 md:px-12 lg:px-16 flex items-center justify-between">
          {/* Brand Wordmark on Left (Matches Image 1) */}
          <a
            href="#"
            className="group inline-flex items-center gap-3.5 text-white transition-opacity hover:opacity-90"
          >
            <div className="font-serif text-2xl md:text-[26px] tracking-tight leading-[1.05] text-left">
              <div>Rotten and Fresh Food</div>
              <div>Checker</div>
            </div>
            <span className="hidden sm:inline-block text-[10px] tracking-[0.22em] uppercase font-sans font-medium px-2 py-0.5 border border-white/40 text-white/90">
              AI VISION
            </span>
          </a>

          {/* Desktop Navigation Links */}
          <nav className="hidden md:flex items-center gap-7 lg:gap-10">
            {navLinks.map((link) => (
              <a
                key={link.label}
                href={link.href}
                className="text-[14px] font-medium tracking-wide text-white/90 hover:text-white transition-colors relative py-1 after:content-[''] after:absolute after:bottom-0 after:left-0 after:w-0 after:h-[1px] after:bg-white hover:after:w-full after:transition-all after:duration-300"
              >
                {link.label}
              </a>
            ))}

            {/* Camera Scan Solid White Button (Matches Image 1) */}
            <button
              onClick={handleCameraScanClick}
              className="premium-button text-[14px] font-medium tracking-wide px-5 py-2.5 bg-white text-[#0f172a] rounded-none border border-white hover:bg-white/90 transition-all duration-300 inline-flex items-center gap-2 shadow-lg"
            >
              <Camera className="w-4 h-4 text-black" />
              <span>Camera Scan</span>
            </button>

            {/* Contact Lab Text Button */}
            <button
              onClick={onOpenInquiry}
              className="text-[14px] font-medium tracking-wide text-white/90 hover:text-white transition-colors"
            >
              Contact Lab
            </button>
          </nav>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            aria-label="Toggle Navigation Menu"
            className="md:hidden p-2 text-white"
          >
            {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>
      </header>

      {/* Mobile Menu Overlay */}
      <div
        className={`fixed inset-0 z-40 bg-[#0f172a]/95 backdrop-blur-xl md:hidden transition-all duration-500 ease-in-out flex flex-col justify-between p-8 pt-28 ${
          mobileMenuOpen ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'
        }`}
      >
        <div className="flex flex-col gap-6">
          <span className="text-[11px] font-sans uppercase tracking-[0.25em] text-white/50 mb-2">
            Navigation Index
          </span>
          {navLinks.map((link, idx) => (
            <a
              key={link.label}
              href={link.href}
              onClick={() => setMobileMenuOpen(false)}
              className="text-white text-3xl font-serif tracking-tight flex items-baseline justify-between border-b border-white/10 pb-4"
            >
              <span>{link.label}</span>
              <span className="text-xs font-sans text-white/40 tracking-widest font-normal">
                0{idx + 1}
              </span>
            </a>
          ))}
        </div>

        <div className="space-y-4 pt-6 border-t border-white/10">
          <button
            onClick={() => {
              setMobileMenuOpen(false);
              handleCameraScanClick();
            }}
            className="w-full bg-white text-[#0f172a] py-3.5 text-center text-sm font-medium tracking-wider uppercase inline-flex items-center justify-center gap-2"
          >
            <Camera className="w-4 h-4" />
            <span>Launch Camera Scanner</span>
          </button>
        </div>
      </div>
    </>
  );
};
