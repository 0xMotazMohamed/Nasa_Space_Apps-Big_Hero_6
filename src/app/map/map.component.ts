import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { LeafletModule } from '@bluehalo/ngx-leaflet';
import { LeafletDrawModule } from '@bluehalo/ngx-leaflet-draw';
import * as L from 'leaflet';
import 'leaflet-draw';
import 'leaflet.heat';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { InputTextModule } from 'primeng/inputtext';
import { SliderModule } from 'primeng/slider';

// Type assertion for L.heatLayer
declare module 'leaflet' {
  export function heatLayer(data: number[][], options?: any): any;
}

interface Photo {
  date: string;
  url: string;
}

@Component({
  selector: 'app-map',
  templateUrl: './map.component.html',
  styleUrls: ['./map.component.css'],
  standalone: true,
  imports: [
    LeafletModule,
    LeafletDrawModule,
    CommonModule,
    FormsModule,
    InputTextModule,
    SliderModule
  ]
})
export class MapComponent implements OnInit {
  map: L.Map | undefined;
  drawnItems: L.FeatureGroup = L.featureGroup();
  heatLayer: any;
  forecast: any = null;
  selectedCity: any = null;
  searchQuery: string = '';
  LHeat: any;
  photos: Photo[] = [
    {
      date: "2025-09-16T14:00",
      url: "https://res.cloudinary.com/dgtkksvmr/image/upload/v1759185808/2025-09-16T14:00:00Zno2.png"
    },
    {
      date: "2025-09-16T20:00",
      url: "https://res.cloudinary.com/dgtkksvmr/image/upload/v1759185809/2025-09-16T20:00:00Zno2.png"
    },
    {
      date: "2025-09-16T16:00",
      url: "https://res.cloudinary.com/dgtkksvmr/image/upload/v1759185812/2025-09-16T16:00:00Zno2.png"
    },
    {
      date: "2025-09-16T21:00",
      url: "https://res.cloudinary.com/dgtkksvmr/image/upload/v1759185814/2025-09-16T21:00:00Zno2.png"
    },
    {
      date: "2025-09-16T12:00",
      url: "https://res.cloudinary.com/dgtkksvmr/image/upload/v1759185815/2025-09-16T12:00:00Zno2.png"
    },
    {
      date: "2025-09-16T11:30",
      url: "https://res.cloudinary.com/dgtkksvmr/image/upload/v1759185816/2025-09-16T11:30:00Zno2.png"
    },
    {
      date: "2025-09-16T12:30",
      url: "https://res.cloudinary.com/dgtkksvmr/image/upload/v1759185828/2025-09-16T12:30:00Zno2.png"
    },
    {
      date: "2025-09-16T16:30",
      url: "https://res.cloudinary.com/dgtkksvmr/image/upload/v1759185830/2025-09-16T16:30:00Zno2.png"
    },
    {
      date: "2025-09-16T15:00",
      url: "https://res.cloudinary.com/dgtkksvmr/image/upload/v1759185832/2025-09-16T15:00:00Zno2.png"
    }
  ];
  selectedPhotoIndex: number = 0;
  imageOverlay: L.ImageOverlay | undefined;

  // Define the border coordinates
  private borderBounds: [number, number][] = [
    [14.01, -167.99], // Bottom-left
    [14.01, -13.01],  // Bottom-right
    [72.99, -13.01],  // Top-right
    [72.99, -167.99]  // Top-left
  ];

  options = {
    layers: [
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 18,
        attribution: '© NASA SPACE APPS 2025'
      }),
      L.polygon(this.borderBounds, {
        color: 'var(--p-primary-color)',
        weight: 1,
        opacity: 0.8,
        fillOpacity: 0,
        className: 'custom-polygon'
      })
    ],
    zoom: 3,
    center: L.latLng((25.01 + 72.99) / 2, (-167.99 + -13.01) / 2)
  };

  drawOptions: any = {
    position: 'topright',
    draw: {
      polygon: {
        shapeOptions: {
          className: 'custom-polygon'
        }
      },
      rectangle: false,
      circle: false,
      marker: {
        icon: L.divIcon({
          className: 'custom-marker',
          html: '<div class="marker-icon"></div>',
          iconSize: [20, 20]
        })
      },
      polyline: false,
      circlemarker: false
    },
    edit: {
      featureGroup: this.drawnItems
    }
  };

  constructor(private http: HttpClient) {}

  ngOnInit(): void {
    // Sort photos by date
    this.photos = this.photos.sort((a, b) =>
      new Date(a.date).getTime() - new Date(b.date).getTime()
    );
  }

  onMapReady(map: L.Map) {
    this.map = map;
    this.drawnItems.addTo(map);
    this.map.fitBounds(this.borderBounds);

    // Initialize image overlay with the first photo
    if (this.photos.length > 0) {
      this.updateImageOverlay(this.photos[this.selectedPhotoIndex].url);
    }

    setTimeout(() => {
      map.invalidateSize();
    }, 0);

    map.on('draw:created' as any, (event: any) => {
      const layer = event.layer;
      this.drawnItems.addLayer(layer);

      const popupContent = `
        <div class="custom-popup">
          <h3>${event.layerType === 'marker' ? 'Marker' : 'Polygon'} Details</h3>
          <div class="popup-field">
            <label>Name:</label>
            <input type="text" class="popup-input name-input" placeholder="Enter name">
          </div>
          <div class="popup-field">
            <label>Value:</label>
            <input type="text" class="popup-input value-input" placeholder="Enter value">
          </div>
          <button class="save-btn">Save</button>
        </div>
      `;

      layer.bindPopup(popupContent, {
        className: 'custom-leaflet-popup'
      }).openPopup();

      layer.on('popupopen', () => {
        const popupElement = document.querySelector('.custom-leaflet-popup');
        if (popupElement) {
          const saveButton = popupElement.querySelector('.save-btn') as HTMLButtonElement;
          const nameInput = popupElement.querySelector('.name-input') as HTMLInputElement;
          const valueInput = popupElement.querySelector('.value-input') as HTMLInputElement;

          saveButton?.addEventListener('click', () => {
            const data = {
              type: event.layerType,
              name: nameInput.value,
              value: valueInput.value,
              coordinates: event.layerType === 'marker'
                ? layer.getLatLng()
                : layer.getLatLngs()
            };
            console.log('Saved data:', data);
            layer.closePopup();
          });
        }
      });

      if (event.layerType === 'marker') {
        const latLng = layer.getLatLng();
        console.log('Marker created:', { lat: latLng.lat, lng: latLng.lng });
      } else {
        const bounds = layer.getBounds();
        console.log('Polygon created:', {
          bounds: {
            north: bounds.getNorth(),
            south: bounds.getSouth(),
            east: bounds.getEast(),
            west: bounds.getWest()
          }
        });
      }
    });
  }

  updateImageOverlay(url: string) {
    if (this.map) {
      // Remove existing image overlay if it exists
      if (this.imageOverlay) {
        this.map.removeLayer(this.imageOverlay);
      }
      // Add new image overlay
      this.imageOverlay = L.imageOverlay(url, this.borderBounds, {
        opacity: 0.6,
        attribution: '',
        className: 'image-overlay'
      }).addTo(this.map);
    }
  }

  onSliderChange() {
    if (this.photos.length > 0) {
      this.updateImageOverlay(this.photos[this.selectedPhotoIndex].url);
    }
  }

  formatDate(date: string): string {
    return new Date(date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  searchCities() {
    console.log('Search query:', this.searchQuery);
  }
}
