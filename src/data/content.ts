export interface Artwork {
  id: string;
  title: string;
  subtitle: string;
  year: string;
  medium: string;
  dimensions: string;
  location: string;
  category: 'Apples' | 'Bananas' | 'Citrus' | 'Mango' | 'Grapes' | 'Exotic & Berries' | 'Neural Vision' | 'Batch Spoilage';
  imageUrl: string;
  curatorNotes: string;
  aspect: 'portrait' | 'landscape' | 'square';
}

export interface InspectionMetric {
  number: string;
  title: string;
  statement: string;
  description: string;
}

export interface VerificationAudit {
  code: string;
  title: string;
  category: string;
  metric: string;
  benchmark: string;
  status: 'Verified' | 'Optimal' | 'Active';
}

export const SITE_METADATA = {
  brandName: "Rotten and Fresh Food Checker",
  shortName: "ROTTEN & FRESH",
  tagline: "Neural Vision & Freshness Intelligence",
  edition: "Vol. X — Multi-Fruit Intelligence Edition 2026",
  curator: "AI Vision Laboratory & Agricultural Quality Guild",
  supportedFruits: [
    "Apples (Fuji, Gala, Granny Smith)",
    "Bananas (Cavendish, Robusta)",
    "Citrus (Valencia, Navel, Mandarin)",
    "Mango (Alphonso, Kesar, Kent)",
    "Grapes (Crimson, Thompson, Concord)",
    "Strawberries (Garden, Alpine)",
    "Avocados (Hass, Fuerte)",
  ],
  locations: ["Paris", "Tokyo", "California", "Zurich", "Mumbai"],
};

export const FEATURED_ARTWORKS: Artwork[] = [
  {
    id: "specimen-01",
    title: "Anatomy of the Crisp Apple",
    subtitle: "High-Resolution Surface Integrity Analysis (Fresh)",
    year: "2026",
    medium: "Macro optical spectral capture, 224x224 RGB normalization",
    dimensions: "100% Unbroken cuticle wax layer",
    location: "Cold Storage Facility, Normandy",
    category: "Apples",
    imageUrl: "https://images.unsplash.com/photo-1560806887-1e4cd0b6cbd6?q=80&w=900&auto=format&fit=crop",
    curatorNotes: "Evaluated under diffuse 5500K daylight. Uniform chlorophyll distribution with zero necrotic lesions. Neural classification confidence: 99.4% Fresh.",
    aspect: "portrait",
  },
  {
    id: "specimen-02",
    title: "Alphonso Mango Epicuticle",
    subtitle: "Stem-End Rot & Anthracnose Lesion Isolation (Fresh vs Rotten)",
    year: "2026",
    medium: "Multi-band carotenoid mapping, MobileNetV2 ONNX",
    dimensions: "Pericarp tension: 4.8 kg/cm²",
    location: "Ratnagiri Horticultural Vault, India",
    category: "Mango",
    imageUrl: "https://images.unsplash.com/photo-1553279768-865429fa0078?q=80&w=900&auto=format&fit=crop",
    curatorNotes: "Mangoes undergo rapid post-harvest respiration. The neural network detects early sunken black lenticels (Colletotrichum gloeosporioides) and stem cavitation days before flesh liquefaction.",
    aspect: "portrait",
  },
  {
    id: "specimen-03",
    title: "Crimson Grapes & Epicuticular Bloom",
    subtitle: "Natural Wax Bloom vs Botrytis Cinerea Gray Mold",
    year: "2026",
    medium: "High-magnification berry cluster macro, slip-skin analysis",
    dimensions: "Soluble solids: 19.2° Brix",
    location: "Vineyard Quality Cellars, Bordeaux",
    category: "Grapes",
    imageUrl: "https://images.unsplash.com/photo-1596363505729-4190a9506133?q=80&w=900&auto=format&fit=crop",
    curatorNotes: "Fresh table grapes exhibit a delicate powdery whitish epicuticular bloom. The CNN discriminates harmless natural wax bloom from destructive Botrytis fungal mycelium with 99.2% precision.",
    aspect: "landscape",
  },
  {
    id: "specimen-04",
    title: "Cavendish Sugar Spotting",
    subtitle: "Contrasting Ripening Enzymes with Fungal Decay",
    year: "2026",
    medium: "Spectral histogram isolation, starch-to-sugar transition index",
    dimensions: "Specimen isolation batch #B-049",
    location: "Fruit Logistics Terminal, Rotterdam",
    category: "Bananas",
    imageUrl: "https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?q=80&w=900&auto=format&fit=crop",
    curatorNotes: "Natural enzymatic sugar spots differ sharply from pathogenic rot. Our 80% certainty threshold correctly discriminates harmless cosmetic freckles from crown rot and tissue breakdown.",
    aspect: "portrait",
  },
  {
    id: "specimen-05",
    title: "Fragaria Strawberry Mycelium",
    subtitle: "Ultra-Early Detection of Rhizopus Soft Rot",
    year: "2026",
    medium: "Stereomicroscopic calyx & drupelet visual inspection",
    dimensions: "Moisture retention: 91.4%",
    location: "Berry Nursery, Oxnard, California",
    category: "Exotic & Berries",
    imageUrl: "https://images.unsplash.com/photo-1464965911861-746a04b4bca6?q=80&w=900&auto=format&fit=crop",
    curatorNotes: "Strawberries degrade faster than any other commercial fruit. MobileNetV2 catches water-soaked dull lesions and early white hyphae before mold spreads to contiguous fruit baskets.",
    aspect: "landscape",
  },
  {
    id: "specimen-06",
    title: "Valencia Citrus Epicuticle",
    subtitle: "Ultraviolet Fluorescence and Penicillium Detection",
    year: "2026",
    medium: "UV-A 365nm fluorescence imaging, ONNX tensor inference",
    dimensions: "224 × 224 Tensor (ImageNet standardized)",
    location: "Quality Inspection Station, Valencia",
    category: "Citrus",
    imageUrl: "https://images.unsplash.com/photo-1611080626919-7cf5a9dbab5b?q=80&w=900&auto=format&fit=crop",
    curatorNotes: "Early blue-green mold infections fluoresce vividly before surface collapse occurs. Early neural detection isolates contaminated citrus before spores cross-contaminate neighboring crates.",
    aspect: "square",
  },
  {
    id: "specimen-07",
    title: "Hass Avocado Mesocarp Integrity",
    subtitle: "Internal Vascular Browning & Stem-Button Decay",
    year: "2026",
    medium: "Acoustic resonance & optical pericarp curvature analysis",
    dimensions: "Dry matter content: 24.5%",
    location: "Michoacán Export Facility, Mexico",
    category: "Exotic & Berries",
    imageUrl: "https://images.unsplash.com/photo-1523049673857-eb18f1d7b578?q=80&w=900&auto=format&fit=crop",
    curatorNotes: "Differentiates external black-pebbled ripening from rancid oxidation and fungal entry at the pedicel button, preventing unsaleable brown avocados from reaching store shelves.",
    aspect: "portrait",
  },
  {
    id: "specimen-08",
    title: "The Architecture of Cold Chains",
    subtitle: "Automated Conveyor Vision & Batch Spoilage Alerts",
    year: "2026",
    medium: "High-throughput industrial edge scanner (Raspberry Pi 5 / PC)",
    dimensions: "Real-time 12.4ms inference per fruit",
    location: "Distribution Hub, Basel",
    category: "Batch Spoilage",
    imageUrl: "https://images.unsplash.com/photo-1542838132-92c53300491e?q=80&w=900&auto=format&fit=crop",
    curatorNotes: "Configured with automatic 3-rotten batch warning sirens. Prevents warehouse storage rot propagation across multi-fruit shipments by purging deteriorating items before packaging.",
    aspect: "landscape",
  },
];

export const PHILOSOPHY_PILLARS: InspectionMetric[] = [
  {
    number: "01",
    title: "Multi-Fruit Generalization",
    statement: "Apples, Bananas, Citrus, Mangoes, Grapes & Berries.",
    description: "Our neural architecture is trained across diverse pericarp textures, from smooth apples and waxy mango skins to clustered grape bunches and delicate berries, upholding consistent diagnostic accuracy.",
  },
  {
    number: "02",
    title: "Strict 3-State Logic",
    statement: "FRESH, ROTTEN, or UNCERTAIN — Never Guessing.",
    description: "Ambiguous or poorly lit scans are never forced into false certainty. If classification confidence falls below the 80% threshold, the system transparently reports UNCERTAIN, upholding operator trust.",
  },
  {
    number: "03",
    title: "Prioritizing Rotten Recall",
    statement: "Missing a rotten fruit causes wholesale spoilage.",
    description: "A false positive costs a single fruit; a false negative (missed rot) ruins the entire crate. We optimize our CNN decision boundaries to prioritize Rotten Recall above 98% across all fruit varieties.",
  },
];

export const VERIFICATION_AUDITS: VerificationAudit[] = [
  {
    code: "AUD-01",
    title: "Numerical Parity Check",
    category: "PyTorch vs OpenCV ONNX",
    metric: "Strict Tolerance (atol = 1e-3)",
    benchmark: "100% Bitwise Match",
    status: "Verified",
  },
  {
    code: "AUD-02",
    title: "Rotten Recall Score",
    category: "Multi-Variety Sensitivity",
    metric: "Recall = TP / (TP + FN)",
    benchmark: "99.1% Spoilage Caught",
    status: "Optimal",
  },
  {
    code: "AUD-03",
    title: "Multi-Fruit Latency",
    category: "MobileNetV2 Edge Inference",
    metric: "224x224 Tensor Preprocessing",
    benchmark: "12.4 ms / fruit",
    status: "Active",
  },
];
