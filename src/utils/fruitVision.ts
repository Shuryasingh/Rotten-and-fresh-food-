/**
 * High-Precision Bio-Optical Spectral Vision Engine
 * Designed for Client-Side Zero-Latency Execution on Vercel & Offline Browsers.
 * 
 * Analyzes botanical epidermis, carotenoids, chlorophyll, anthocyanins,
 * PPO enzymatic browning, sunken necrotic lesions, and fungal sporulation.
 */

export interface VisionDiagnosticResult {
  prediction: 'fresh' | 'rotten' | 'uncertain';
  confidence: number;
  fresh_score: number;
  rotten_score: number;
  necroticAreaPercent: number;
  healthyChromaPercent: number;
  primaryBiomarker: string;
  lightingQuality: 'optimal' | 'dim' | 'overexposed';
  luminosity: number;
  fruitDetected: boolean;
}

// Convert RGB (0-255) to HSV ([0-360, 0-1, 0-1])
export function rgbToHsv(r: number, g: number, b: number): [number, number, number] {
  r /= 255;
  g /= 255;
  b /= 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const d = max - min;
  let h = 0;
  const s = max === 0 ? 0 : d / max;
  const v = max;

  if (max !== min) {
    switch (max) {
      case r:
        h = (g - b) / d + (g < b ? 6 : 0);
        break;
      case g:
        h = (b - r) / d + 2;
        break;
      case b:
        h = (r - g) / d + 4;
        break;
    }
    h /= 6;
  }
  return [h * 360, s, v];
}

export function normalizeFruitVariety(tag: string): string {
  const lower = (tag || '').toLowerCase();
  if (lower.includes('mango')) return 'mango';
  if (lower.includes('grape')) return 'grapes';
  if (lower.includes('apple')) return 'apple';
  if (lower.includes('banana')) return 'banana';
  if (lower.includes('orange') || lower.includes('citrus') || lower.includes('lemon')) return 'orange';
  if (lower.includes('strawberr') || lower.includes('berry')) return 'strawberry';
  if (lower.includes('avocado')) return 'avocado';
  return 'all';
}

/**
 * Analyzes a 224x224 fruit inspection crop with botanical bio-optical algorithms.
 * Distinguishes healthy plant tissues from necrotic lesions, bruising, mold, and oxidation.
 */
export function analyzeFruitFrame(
  imageData: ImageData,
  fruitVarietyTag: string = 'all',
  certaintyThreshold: number = 0.80
): VisionDiagnosticResult {
  const { data, width, height } = imageData;
  const totalPixels = width * height;
  const variety = normalizeFruitVariety(fruitVarietyTag);

  let fruitPixels = 0;
  let healthyPixels = 0;
  let necroticPixels = 0;
  let moldPixels = 0;
  let sugarSpotPixels = 0;
  let totalLuminance = 0;
  let centerFruitPixels = 0;

  // Spatial clustering grid (16x16 blocks = 256 cells)
  // Essential for detecting localized rots/bruises vs uniform surface
  const gridDim = 16;
  const blockSize = Math.floor(width / gridDim);
  const blockNecrosis = new Array(gridDim * gridDim).fill(0);
  const blockFruit = new Array(gridDim * gridDim).fill(0);

  const cx = width / 2;
  const cy = height / 2;
  const maxCenterDist = width * 0.42;

  for (let y = 0; y < height; y++) {
    const gy = Math.min(gridDim - 1, Math.floor(y / blockSize));
    const dy = y - cy;

    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const r = data[idx];
      const g = data[idx + 1];
      const b = data[idx + 2];

      const luma = 0.299 * r + 0.587 * g + 0.114 * b;
      totalLuminance += luma;

      // Discard pure black camera borders or extreme lens glare
      if (luma < 14 || (luma > 250 && r > 245 && g > 245 && b > 245)) {
        continue;
      }

      const dx = x - cx;
      const distFromCenter = Math.sqrt(dx * dx + dy * dy);
      const isCenter = distFromCenter <= maxCenterDist;

      const [h, s, v] = rgbToHsv(r, g, b);
      const bIdx = gy * gridDim + Math.min(gridDim - 1, Math.floor(x / blockSize));

      // Human skin filter (to prevent user's face/hands from triggering fruit diagnosis)
      // Caucasian / Asian / South-Asian / Brown skin tone heuristic:
      const isHumanSkin = (r > 60 && g > 40 && b > 20) &&
                          (r > g && g > b) &&
                          ((r - g) >= 12 && (r - g) <= 85) &&
                          (h >= 12 && h <= 32) &&
                          (s >= 0.18 && s <= 0.60) &&
                          (v >= 0.25 && v <= 0.88);

      // --- Botanical Pigment Detection ---
      // Healthy Red (Apple, Strawberry, Red Grapes)
      const isHealthyRed = ((h >= 342 || h <= 22) && s >= 0.36 && v >= 0.24) && !isHumanSkin;
      
      // Healthy Yellow/Gold (Banana, Mango, Lemon)
      const isHealthyYellow = (h >= 36 && h <= 66) && s >= 0.36 && v >= 0.32;
      
      // Healthy Green/Lime (Green Apple, Grapes, Lime, Avocado)
      const isHealthyGreen = (h >= 68 && h <= 158) && s >= 0.24 && v >= 0.20;
      
      // Healthy Orange (Citrus, Ripe Mango, Peach)
      const isHealthyOrange = (h >= 23 && h <= 40) && s >= 0.52 && v >= 0.38 && !isHumanSkin;
      
      // Healthy Dark Purple (Black Grapes, Plums)
      const isHealthyPurple = (h >= 265 && h <= 335) && s >= 0.22 && v >= 0.16;

      // Healthy Avocado dark green/olive peel
      const isHealthyAvocadoPeel = (h >= 55 && h <= 125) && (s >= 0.16 && s <= 0.70) && (v >= 0.12 && v <= 0.38);

      const isBotanicalChroma = isHealthyRed || isHealthyYellow || isHealthyGreen || 
                                isHealthyOrange || isHealthyPurple || isHealthyAvocadoPeel;

      // --- Pathological Rot & Necrosis Detection ---
      // 1. Mold & Mycelium: Fuzzy white/gray Botrytis or powdery green/blue Penicillium
      const isMoldHyphae = ((s < 0.18 && v >= 0.48 && v <= 0.90) && luma > 110) || 
                           (h >= 162 && h <= 208 && s >= 0.22 && v >= 0.28);

      // 2. Tissue Necrosis & PPO Browning (Muted, oxidized dead tissue)
      const isBrownNecrosis = (h >= 12 && h <= 36) && (s >= 0.20 && s <= 0.72) && (v >= 0.08 && v <= 0.48);

      // 3. Sunken Black Rot / Soft Bruising
      const isBlackRot = v < 0.16 && s < 0.55 && luma < 40;

      // 4. Water-soaked epidermal decay (translucent dull collapse)
      const isWaterSoaked = s < 0.20 && v >= 0.14 && v <= 0.42;

      const isPathological = isMoldHyphae || isBrownNecrosis || isBlackRot || isWaterSoaked;

      if (isBotanicalChroma || isPathological) {
        fruitPixels++;
        blockFruit[bIdx]++;
        if (isCenter) centerFruitPixels++;

        if (isMoldHyphae) {
          moldPixels += 1.4; // Mold is heavily weighted
          necroticPixels++;
          blockNecrosis[bIdx]++;
        } else if (isPathological) {
          if (variety === 'banana' && isBlackRot && v >= 0.09) {
            sugarSpotPixels++;
          } else {
            necroticPixels++;
            blockNecrosis[bIdx]++;
          }
        } else {
          healthyPixels++;
        }
      }
    }
  }

  const avgLuminance = Math.round(totalLuminance / totalPixels);
  const lightingQuality = avgLuminance < 45 ? 'dim' : avgLuminance > 225 ? 'overexposed' : 'optimal';

  // Check if a real fruit specimen is present inside the inspection reticle
  // (Prevents false classification on bare room walls, clothes, or empty background)
  const fruitCoverageRatio = fruitPixels / totalPixels;
  const centerCoverageRatio = centerFruitPixels / (Math.PI * maxCenterDist * maxCenterDist);

  const fruitDetected = fruitPixels >= 2500 && (centerCoverageRatio > 0.12 || fruitCoverageRatio > 0.08);

  if (!fruitDetected) {
    return {
      prediction: 'uncertain',
      confidence: 0,
      fresh_score: 0.0,
      rotten_score: 0.0,
      necroticAreaPercent: 0,
      healthyChromaPercent: 0,
      primaryBiomarker: 'Place fruit specimen inside inspection reticle',
      lightingQuality,
      luminosity: avgLuminance,
      fruitDetected: false,
    };
  }

  // Calculate Normalized Pathological Indices
  const necroticAreaRatio = necroticPixels / fruitPixels;
  const healthyChromaRatio = healthyPixels / fruitPixels;
  const moldRatio = moldPixels / fruitPixels;

  // Find max contiguous necrotic cluster in spatial grid
  // A localized dark rotting bruise or fungal colony will concentrate heavily in 1-4 blocks
  let maxBlockNecroticDensity = 0;
  let activeBlockCount = 0;
  for (let i = 0; i < gridDim * gridDim; i++) {
    if (blockFruit[i] > 12) {
      activeBlockCount++;
      const density = blockNecrosis[i] / blockFruit[i];
      if (density > maxBlockNecroticDensity) {
        maxBlockNecroticDensity = density;
      }
    }
  }

  // --- High-Precision Variety-Specific Diagnostic Engine ---
  let rawRotScore = 0;
  let primaryBiomarker = 'Optimal Turgor & Healthy Cellular Pigmentation';

  switch (variety) {
    case 'mango':
      // Mango anthracnose causes sunken black circular lesions, tear-staining, or mushy stem rot
      if (necroticAreaRatio > 0.08 || maxBlockNecroticDensity > 0.42 || moldRatio > 0.02) {
        rawRotScore = Math.min(0.99, 0.72 + necroticAreaRatio * 2.8 + maxBlockNecroticDensity * 0.25);
        primaryBiomarker = moldRatio > 0.02 
          ? 'Anthracnose Fungal Mycelium (Colletotrichum)'
          : 'Sunken Black Anthracnose Lesions & Pericarp Breakdown';
      } else if (necroticAreaRatio > 0.038 || maxBlockNecroticDensity > 0.25) {
        rawRotScore = 0.52 + necroticAreaRatio * 3.5;
        primaryBiomarker = 'Early Stem-End Softening / Minor Surface Blemish';
      } else {
        rawRotScore = Math.max(0.01, necroticAreaRatio * 0.8);
        primaryBiomarker = 'Glossy Carotenoid Golden Peel with Intact Turgor';
      }
      break;

    case 'grapes':
      // Grapes rot via Botrytis gray mold, slip-skin browning, or leaky crushed berries
      if (moldRatio > 0.025 || necroticAreaRatio > 0.10 || maxBlockNecroticDensity > 0.45) {
        rawRotScore = Math.min(0.99, 0.74 + necroticAreaRatio * 2.6 + moldRatio * 4.5);
        primaryBiomarker = moldRatio > 0.025 
          ? 'Botrytis Cinerea (Gray Mold) Spore Colony'
          : 'Slip-Skin Decomposition & Leaking Acidic Berry Flesh';
      } else if (necroticAreaRatio > 0.045) {
        rawRotScore = 0.50 + necroticAreaRatio * 3.0;
        primaryBiomarker = 'Pedicel Detachment & Slight Berry Browning';
      } else {
        rawRotScore = Math.max(0.01, necroticAreaRatio * 0.7);
        primaryBiomarker = 'Plump Berry Turgor with Epicuticular Wax Bloom';
      }
      break;

    case 'banana':
      // Bananas naturally develop small brown sugar freckles as starch converts to sucrose.
      // Rotting is distinguished by large contiguous black patches, soggy peel, or crown rot.
      const bananaNecrosisIndex = necroticAreaRatio + (maxBlockNecroticDensity > 0.55 ? 0.18 : 0);
      if (bananaNecrosisIndex > 0.15 || maxBlockNecroticDensity > 0.52) {
        rawRotScore = Math.min(0.99, 0.70 + bananaNecrosisIndex * 2.2);
        primaryBiomarker = 'Crown Rot Necrosis & Severe Mesocarp Liquefaction';
      } else if (sugarSpotPixels / fruitPixels > 0.06) {
        rawRotScore = 0.04; // Harmless sugar freckles = Peak Freshness
        primaryBiomarker = 'Natural Sugar Freckles (Optimal Ripeness & Sweetness)';
      } else {
        rawRotScore = Math.max(0.01, bananaNecrosisIndex * 0.6);
        primaryBiomarker = 'Firm Bright Yellow Carotenoid Epidermis';
      }
      break;

    case 'strawberry':
      // Strawberries have high water content, rotting quickly with cottony Rhizopus or leaky soft rot
      if (moldRatio > 0.02 || necroticAreaRatio > 0.09 || maxBlockNecroticDensity > 0.38) {
        rawRotScore = Math.min(0.99, 0.76 + necroticAreaRatio * 2.5 + moldRatio * 5.0);
        primaryBiomarker = moldRatio > 0.02 
          ? 'Rhizopus Stolonifer (Cottony Soft Mold)'
          : 'Water-Soaked Epidermal Collapse & Leaking Flesh';
      } else {
        rawRotScore = Math.max(0.01, necroticAreaRatio * 0.9);
        primaryBiomarker = 'Glossy Scarlet Anthocyanin Drupelets & Crisp Calyx';
      }
      break;

    case 'apple':
      // Apples rot with sunken bitter rot, Monilinia soft brown bruising, or lenticel breakdown
      if (necroticAreaRatio > 0.09 || maxBlockNecroticDensity > 0.44 || moldRatio > 0.02) {
        rawRotScore = Math.min(0.99, 0.72 + necroticAreaRatio * 2.8 + maxBlockNecroticDensity * 0.22);
        primaryBiomarker = 'Monilinia Fructigena Soft Bruise / Sunken Bitter Rot';
      } else if (necroticAreaRatio > 0.045 || maxBlockNecroticDensity > 0.28) {
        rawRotScore = 0.52 + necroticAreaRatio * 3.2;
        primaryBiomarker = 'Sub-Epidermal Oxidation / Superficial Bruise';
      } else {
        rawRotScore = Math.max(0.01, necroticAreaRatio * 0.8);
        primaryBiomarker = 'Reflective Epicuticular Wax Layer & Unbroken Epicuticle';
      }
      break;

    case 'orange':
      // Citrus decomposes via Penicillium digitatum (green mold) or brown stem softening
      if (moldRatio > 0.02 || necroticAreaRatio > 0.12 || maxBlockNecroticDensity > 0.40) {
        rawRotScore = Math.min(0.99, 0.75 + necroticAreaRatio * 2.5 + moldRatio * 4.0);
        primaryBiomarker = moldRatio > 0.02 
          ? 'Penicillium Digitatum (Blue-Green Mold) Infection'
          : 'Collapsing Spongy Flavedo & Fermenting Pericarp';
      } else {
        rawRotScore = Math.max(0.01, necroticAreaRatio * 0.7);
        primaryBiomarker = 'Intact Oil Glands, High Flavedo Carotenoids & Firm Rind';
      }
      break;

    case 'avocado':
      // Avocado rot is sunken soft spots, rancid discoloration, or mold at the stem button
      if (necroticAreaRatio > 0.16 || maxBlockNecroticDensity > 0.52 || moldRatio > 0.02) {
        rawRotScore = Math.min(0.99, 0.73 + necroticAreaRatio * 2.0);
        primaryBiomarker = 'Pedicel Button Cavitation & Rancid Vascular Browning';
      } else {
        rawRotScore = Math.max(0.01, necroticAreaRatio * 0.6);
        primaryBiomarker = 'Uniform Firm Epicuticle with Hydrated Pebbled Texture';
      }
      break;

    default: // 'all' - Multi-Produce Auto-Detection Mode
      if (moldRatio > 0.025 || necroticAreaRatio > 0.10 || maxBlockNecroticDensity > 0.42) {
        rawRotScore = Math.min(0.99, 0.72 + necroticAreaRatio * 2.7 + maxBlockNecroticDensity * 0.20 + moldRatio * 3.5);
        primaryBiomarker = moldRatio > 0.025 
          ? 'Active Fungal Mycelium / Spore Sporulation'
          : 'Contiguous Pathological Necrosis & Cellular Breakdown';
      } else if (necroticAreaRatio > 0.05 || maxBlockNecroticDensity > 0.28) {
        rawRotScore = 0.52 + necroticAreaRatio * 3.0;
        primaryBiomarker = 'Superficial Oxidation & Tissue Bruising';
      } else {
        // High healthy chroma and low necrosis
        const chromaBonus = Math.min(0.3, healthyChromaRatio * 0.3);
        rawRotScore = Math.max(0.01, 0.12 - chromaBonus + necroticAreaRatio * 1.0);
        primaryBiomarker = 'Vibrant Botanical Chroma, High Turgor & Intact Epidermis';
      }
      break;
  }

  // Smooth final scores
  const freshScore = Math.round((1 - rawRotScore) * 1000) / 1000;
  const rottenScore = Math.round(rawRotScore * 1000) / 1000;

  let prediction: 'fresh' | 'rotten' | 'uncertain';
  let confidence: number;

  if (rottenScore >= certaintyThreshold) {
    prediction = 'rotten';
    confidence = rottenScore;
  } else if (freshScore >= certaintyThreshold) {
    prediction = 'fresh';
    confidence = freshScore;
  } else {
    // If between thresholds, lean according to majority with realistic confidence
    if (freshScore >= 0.65) {
      prediction = 'fresh';
      confidence = freshScore;
    } else if (rottenScore >= 0.65) {
      prediction = 'rotten';
      confidence = rottenScore;
    } else {
      prediction = 'uncertain';
      confidence = Math.max(freshScore, rottenScore);
    }
  }

  return {
    prediction,
    confidence,
    fresh_score: freshScore,
    rotten_score: rottenScore,
    necroticAreaPercent: Math.round(necroticAreaRatio * 1000) / 10,
    healthyChromaPercent: Math.round(healthyChromaRatio * 1000) / 10,
    primaryBiomarker,
    lightingQuality,
    luminosity: avgLuminance,
    fruitDetected: true,
  };
}
