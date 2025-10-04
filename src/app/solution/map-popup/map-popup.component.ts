import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { TabsModule } from 'primeng/tabs';
import { CityResponse } from '../../shared/interfaces/CityResponse';
import { PointsResponse } from '../../shared/interfaces/PointsResponse';
import { PolygonResponse } from '../../shared/interfaces/PolygonResponse';

type AnyAQI = CityResponse | PointsResponse | PolygonResponse;
type Reading = {
  date: string;
  no2?: { value: number; AQI: { value: number; category: string } };
  o3?: { value: number; AQI: { value: number; category: string } };
  hcho?: { value: number; AQI: { value: number; category: string } };
  AQI_General?: { value: number; category: string };
};

@Component({
  selector: 'app-map-popup',
  standalone: true,
  imports: [CommonModule, TabsModule],
  templateUrl: './map-popup.component.html',
  styleUrls: ['./map-popup.component.css']
})
export class MapPopupComponent {
  public _aqiData!: AnyAQI;

  @Input() set aqiData(v: AnyAQI) {
    this._aqiData = v;
    this.normalize();
  }
  get aqiData(): AnyAQI { return this._aqiData; }

  name = 'Unknown';
  dates: string[] = [];
  series: Reading[] = [];
  activeTabIndex = 0;
  city:any;
  polygon:any;
  point:any;

  ngOnInit(){
    this.dates = this._aqiData.dates;
    console.log(this._aqiData);

  }

  private normalize() {
    if (!this._aqiData) {
      this.name = 'Unknown';
      this.dates = [];
      this.series = [];
      return;
    }

    this.dates = Array.isArray((this._aqiData as any)?.dates) ? (this._aqiData as any).dates : [];
    this.series = [];
    this.name = 'Unknown';

    const hasCity = (d: AnyAQI): d is CityResponse | PointsResponse => 'city' in d && !!(d as any).city;
    const hasPoint = (d: AnyAQI): d is PointsResponse => 'point' in d && !!(d as any).point;
    const hasPolygon = (d: AnyAQI): d is PolygonResponse => 'polygon' in d && !!(d as any).polygon;

    if (hasCity(this._aqiData) && (this._aqiData as any).city?.values) {
      this.name = (this._aqiData as any).city?.name ?? 'Selected City';
      this.series = (this._aqiData as any).city?.values ?? [];
    } else if (hasPoint(this._aqiData)) {
      this.name = 'Selected Point';
      this.series = this._aqiData.point?.values ?? [];
    } else if (hasPolygon(this._aqiData)) {
      this.name = 'Selected Area';
      this.series = this._aqiData.polygon?.values ?? [];
    }

    if (!this.dates?.length && this.series?.length) {
      this.dates = this.series.map(r => r.date);
    }

    if (this.activeTabIndex >= this.dates.length) this.activeTabIndex = 0;
  }

  getAQIColor(aqiValue: number | undefined): string {
    if (aqiValue === undefined || aqiValue === null) return 'var(--app-fg)'; // Default to text color if N/A

    if (aqiValue <= 50) return '#00E400';    // Good
    if (aqiValue <= 100) return '#FFFF00';   // Moderate
    if (aqiValue <= 150) return '#FF7E00';   // Unhealthy for Sensitive Groups
    if (aqiValue <= 200) return '#FF0000';   // Unhealthy
    if (aqiValue <= 300) return '#8F3F97';   // Very Unhealthy
    return '#7E0023';                        // Hazardous
  }

  formatDate(date: string): string {
    return new Date(date).toLocaleString('en-US', {
      day: '2-digit',
      month: 'short',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  }
  ngAfterViewInit(){
    console.log(this._aqiData);

  }
}
