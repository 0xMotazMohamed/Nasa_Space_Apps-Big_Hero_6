import { HttpClient, HttpParams } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, of } from 'rxjs';
import { CityResponse } from '../../shared/interfaces/CityResponse';
import { LayerResponse } from '../../shared/interfaces/LayerResponse';
import { PointsResponse } from '../../shared/interfaces/PointsResponse';
import { PolygonResponse } from '../../shared/interfaces/PolygonResponse';

@Injectable({
  providedIn: 'root'
})
export class MapService {
  private http = inject(HttpClient);
  apiUrl = '';

  constructor() { }

  getPoints(lon: string, lat: string): Observable<PointsResponse> {
    if (+lon >= -13.01 || +lon <= -167.99 || +lat >= 72.99 || +lat <= 14.01) {
      return of({} as PointsResponse);
    } else {
      let params = new HttpParams()
        .set('lon', lon)
        .set('lat', lat);
      return this.http.get<PointsResponse>(`${this.apiUrl}/points`, { params });
    }
  }

  getLayer(layer_type: string): Observable<LayerResponse> {
    return this.http.get<LayerResponse>(`${this.apiUrl}/layer/${layer_type}`);
  }

  getCity(city_id: string): Observable<CityResponse> {
    let params = new HttpParams()
      .set('city_id', city_id);
    return this.http.get<CityResponse>(`${this.apiUrl}/city`, { params });
  }

  getPolygonData(coordinates: any): Observable<PolygonResponse> {
    let params = new HttpParams()
      .set('coordinates', coordinates);
    return this.http.get<PolygonResponse>(`${this.apiUrl}/get_polygon`, { params });
  }
}
