import React, { useState, useEffect } from 'react';
import { X, Check } from 'lucide-react';

interface InquiryModalProps {
  isOpen: boolean;
  onClose: () => void;
  defaultSubject?: string;
}

export const InquiryModal: React.FC<InquiryModalProps> = ({
  isOpen,
  onClose,
  defaultSubject = 'Private Viewing & Commission',
}) => {
  const [submitted, setSubmitted] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    organization: '',
    inquiryType: 'Private Viewing',
    message: '',
  });

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) {
      document.body.style.overflow = 'hidden';
      window.addEventListener('keydown', handleKeyDown);
    }
    return () => {
      document.body.style.overflow = '';
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitted(true);
    setTimeout(() => {
      setSubmitted(false);
      onClose();
    }, 2800);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-[#0f172a]/80 backdrop-blur-md animate-fade-rise">
      <div
        className="relative bg-[#ffffff] w-full max-w-xl p-8 sm:p-12 text-[#0f172a] shadow-2xl border border-[rgba(15,23,42,0.1)] space-y-8"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close Button */}
        <button
          onClick={onClose}
          aria-label="Close dialog"
          className="absolute top-6 right-6 p-2 hover:bg-black hover:text-white transition-colors"
        >
          <X className="w-5 h-5" />
        </button>

        {submitted ? (
          <div className="py-12 text-center space-y-4">
            <div className="w-12 h-12 bg-black text-white rounded-full flex items-center justify-center mx-auto mb-4">
              <Check className="w-6 h-6" />
            </div>
            <h3 className="font-serif text-3xl text-[#0f172a]">
              Dossier Transmitted
            </h3>
            <p className="font-sans text-sm text-[hsl(215,25%,32%)] max-w-md mx-auto leading-relaxed">
              Our curatorial secretariat in Paris will review your request and reply within 24 hours with confidential exhibition catalogs.
            </p>
          </div>
        ) : (
          <>
            <div className="space-y-2">
              <span className="editorial-tag text-[hsl(215,25%,32%)] block">
                Direct Curatorial Inquiry
              </span>
              <h3 className="font-serif text-3xl sm:text-4xl text-[#0f172a]">
                Initiate Private Dialogue
              </h3>
              <p className="font-sans text-xs sm:text-sm text-[hsl(215,25%,32%)]">
                Subject: <span className="font-semibold text-[#0f172a]">{defaultSubject}</span>
              </p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-5 text-xs font-sans">
              <div>
                <label className="block uppercase tracking-wider text-[hsl(215,25%,32%)] mb-2 font-medium">
                  Full Name *
                </label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Lady Vivienne Montgomery"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  className="w-full border border-[rgba(15,23,42,0.2)] px-4 py-3 text-sm focus:outline-none focus:border-black transition-colors"
                />
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label className="block uppercase tracking-wider text-[hsl(215,25%,32%)] mb-2 font-medium">
                    Email Address *
                  </label>
                  <input
                    type="email"
                    required
                    placeholder="curator@institution.art"
                    value={formData.email}
                    onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                    className="w-full border border-[rgba(15,23,42,0.2)] px-4 py-3 text-sm focus:outline-none focus:border-black transition-colors"
                  />
                </div>
                <div>
                  <label className="block uppercase tracking-wider text-[hsl(215,25%,32%)] mb-2 font-medium">
                    Institution / Foundation
                  </label>
                  <input
                    type="text"
                    placeholder="e.g. Fondation Beyeler"
                    value={formData.organization}
                    onChange={(e) => setFormData({ ...formData, organization: e.target.value })}
                    className="w-full border border-[rgba(15,23,42,0.2)] px-4 py-3 text-sm focus:outline-none focus:border-black transition-colors"
                  />
                </div>
              </div>

              <div>
                <label className="block uppercase tracking-wider text-[hsl(215,25%,32%)] mb-2 font-medium">
                  Inquiry Purpose
                </label>
                <select
                  value={formData.inquiryType}
                  onChange={(e) => setFormData({ ...formData, inquiryType: e.target.value })}
                  className="w-full border border-[rgba(15,23,42,0.2)] px-4 py-3 text-sm focus:outline-none focus:border-black bg-white transition-colors"
                >
                  <option value="Private Viewing">Private Viewing (Paris / Kyoto / Basel)</option>
                  <option value="Acquisition">Oeuvre Acquisition</option>
                  <option value="Architectural Commission">Architectural Commission</option>
                  <option value="Institutional Exhibition">Institutional Exhibition Loan</option>
                </select>
              </div>

              <div>
                <label className="block uppercase tracking-wider text-[hsl(215,25%,32%)] mb-2 font-medium">
                  Confidential Message / Intent
                </label>
                <textarea
                  rows={4}
                  placeholder="Please state preferred dates, locations, or specific catalog pieces under consideration..."
                  value={formData.message}
                  onChange={(e) => setFormData({ ...formData, message: e.target.value })}
                  className="w-full border border-[rgba(15,23,42,0.2)] px-4 py-3 text-sm focus:outline-none focus:border-black transition-colors resize-none"
                />
              </div>

              <div className="pt-2">
                <button
                  type="submit"
                  className="premium-button w-full py-4 bg-[#000000] text-[#ffffff] text-xs font-sans font-medium uppercase tracking-[0.18em] border border-black hover:bg-[#1a1a1a] transition-all"
                >
                  Transmit Curatorial Request
                </button>
              </div>
            </form>
          </>
        )}
      </div>
    </div>
  );
};
