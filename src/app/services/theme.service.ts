import { Injectable } from '@angular/core';

const STEPS = [0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950] as const;
type Step = typeof STEPS[number];
type Scale = Record<Step, string>;

type ScalePct = '25%' | '50%' | '75%' | '100%';

@Injectable({ providedIn: 'root' })
export class ThemeService {
  private readonly KEY_ACCENT = 'accentColor';
  private readonly KEY_DARK = 'darkMode';
  private readonly KEY_SCALE = 'uiScale';

  private readonly defaultAccent = '#369FFC';
  private readonly defaultDark = true;        // <- default dark mode ON
  private readonly defaultScale: ScalePct = '100%';

  /** Call this once on app start */
  init() {
    const accent = this.getAccent();
    const dark = this.getDarkMode();
    const scale = this.getScale();

    this.applyAccent(accent);
    this.applyDarkMode(dark);
    this.applyScale(scale);
  }

  // -------- accent color --------
  saveAndApplyAccent(accent: string) {
    const hex = this.normalize(accent);
    localStorage.setItem(this.KEY_ACCENT, hex);
    this.applyAccent(hex);
  }

  applyAccent(accent: string) {
    const scale = this.makeScale(this.normalize(accent));
    const root = document.documentElement.style;

    STEPS.forEach((step) => {
      root.setProperty(`--p-primary-${step}`, scale[step]); // Prime tokens
    });

    root.setProperty('--p-primary-color', scale[500]);
    root.setProperty('--p-primary-contrast', this.getContrast(scale[500]));
  }

  getAccent(): string {
    return localStorage.getItem(this.KEY_ACCENT) || this.defaultAccent;
  }

  // -------- dark mode --------
  setDarkMode(enabled: boolean) {
    localStorage.setItem(this.KEY_DARK, JSON.stringify(enabled));
    this.applyDarkMode(enabled);
  }

  applyDarkMode(enabled: boolean) {
    const root = document.documentElement;
    root.classList.toggle('app-dark', enabled);   // <-- matches '.app-dark' above
    root.style.setProperty('color-scheme', enabled ? 'dark' : 'light');
  }

  getDarkMode(): boolean {
    const raw = localStorage.getItem(this.KEY_DARK);
    return raw === null ? this.defaultDark : JSON.parse(raw);
  }

  // -------- scale (html font-size) --------
  setScale(scale: ScalePct) {
    localStorage.setItem(this.KEY_SCALE, scale);
    this.applyScale(scale);
  }

  applyScale(scale: ScalePct) {
    // 100% → 14px, 75% → 13px, 50% → 12px, 25% → 11px
    const px = this.scaleToPx(scale);
    // Inline style wins over stylesheet: no extra CSS required
    document.documentElement.style.fontSize = px;
    document.documentElement.style.setProperty('--app-font-size', px);

  }

  getScale(): ScalePct {
    const raw = localStorage.getItem(this.KEY_SCALE) as ScalePct | null;
    return raw ?? this.defaultScale;
  }

  private scaleToPx(scale: ScalePct): string {
    switch (scale) {
      case '25%': return '11px';
      case '50%': return '12px';
      case '75%': return '13px';
      case '100%':
      default:    return '14px';
    }
  }

  // -------- helpers --------
  private makeScale(base: string): Scale {
    const white = '#FFFFFF';
    const black = '#000000';
    const b = this.normalize(base);

    return {
      0:   white,
      50:  this.mix(b, white, 0.92),
      100: this.mix(b, white, 0.85),
      200: this.mix(b, white, 0.72),
      300: this.mix(b, white, 0.58),
      400: this.mix(b, white, 0.40),
      500: b,
      600: this.mix(b, black, 0.15),
      700: this.mix(b, black, 0.30),
      800: this.mix(b, black, 0.50),
      900: this.mix(b, black, 0.70),
      950: this.mix(b, black, 0.85),
    };
  }

  private normalize(hex: string) {
    const h = (hex || '').trim();
    return (h.startsWith('#') ? h : `#${h}`).toUpperCase();
  }

  private mix(aHex: string, bHex: string, w: number) {
    const a = this.hexToRgb(aHex);
    const b = this.hexToRgb(bHex);
    const r = Math.round(a.r + (b.r - a.r) * w);
    const g = Math.round(a.g + (b.g - a.g) * w);
    const bl = Math.round(a.b + (b.b - a.b) * w);
    return this.rgbToHex(r, g, bl);
  }

  private hexToRgb(hex: string) {
    const h = hex.replace('#', '');
    const int = parseInt(h, 16);
    return { r: (int >> 16) & 255, g: (int >> 8) & 255, b: int & 255 };
  }

  private rgbToHex(r: number, g: number, b: number) {
    const to = (n: number) => n.toString(16).padStart(2, '0');
    return `#${to(r)}${to(g)}${to(b)}`.toUpperCase();
  }

  private getContrast(hex: string) {
    const { r, g, b } = this.hexToRgb(hex);
    const yiq = (r * 299 + g * 587 + b * 114) / 1000;
    return yiq >= 128 ? '#000000' : '#FFFFFF';
  }
}
