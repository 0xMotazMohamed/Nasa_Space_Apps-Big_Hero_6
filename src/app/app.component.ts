import { Component, inject } from '@angular/core';
import { MapComponent } from "./map/map.component";
import { ThemeService } from './services/theme.service';

@Component({
  selector: 'app-root',
  imports: [MapComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'weather-map';
  private theme = inject(ThemeService);
  ngOnInit(){
    this.theme.init();
  }

}
