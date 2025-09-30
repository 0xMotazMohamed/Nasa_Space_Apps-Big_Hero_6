import { definePreset } from '@primeng/themes';
import Aura from '@primeng/themes/aura';

function hexToRgb(hex: string) {
  const m = hex.replace('#','').match(/^([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i);
  if (!m) return { r: 54, g: 159, b: 252 }; // fallback #369FFC
  return { r: parseInt(m[1], 16), g: parseInt(m[2], 16), b: parseInt(m[3], 16) };
}

function rgbToHex(r: number, g: number, b: number) {
  const toHex = (v: number) => v.toString(16).padStart(2, '0');
  return `#${toHex(Math.round(r))}${toHex(Math.round(g))}${toHex(Math.round(b))}`.toUpperCase();
}

/** Linear mix in sRGB (simple and good enough for UI tokens) */
function mix(hexA: string, hexB: string, t: number) {
  const a = hexToRgb(hexA);
  const b = hexToRgb(hexB);
  const r = a.r + (b.r - a.r) * t;
  const g = a.g + (b.g - a.g) * t;
  const b2 = a.b + (b.b - a.b) * t;
  return rgbToHex(r, g, b2);
}

/** Build a 0–950 scale from a single base color (#RRGGBB). */
export function makeScale(base: string) {
  const white = '#FFFFFF';
  const black = '#000000';
  const baseHex = /^#?[0-9a-f]{6}$/i.test(base) ? (base.startsWith('#') ? base : `#${base}`) : '#369FFC';

  return {
    0:   white,
    50:  mix(baseHex, white, 0.92),
    100: mix(baseHex, white, 0.85),
    200: mix(baseHex, white, 0.72),
    300: mix(baseHex, white, 0.58),
    400: mix(baseHex, white, 0.40),
    500: baseHex,                    // base
    600: mix(baseHex, black, 0.15),
    700: mix(baseHex, black, 0.30),
    800: mix(baseHex, black, 0.50),
    900: mix(baseHex, black, 0.70),
    950: mix(baseHex, black, 0.85),
  };
}

const basePrimary = localStorage.getItem('accentColor') || '#369FFC';
const primaryScale = makeScale(basePrimary);

export const MyPreset = definePreset(Aura, {
  semantic: {
    primary: primaryScale
    },
    secondary: {
      0:   '#FFFFFF',
      50:  '#FFF4EB',
      100: '#FEE9D7',
      200: '#FED4AF',
      300: '#FDBE86',
      400: '#FDA95E',
      500: '#FC9336', // Base
      600: '#E38431',
      700: '#BD6E28',
      800: '#8B511E',
      900: '#583313',
      950: '#321D0B'
    },
  }
);
