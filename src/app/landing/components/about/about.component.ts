import { Component } from '@angular/core';
import { PlanetComponent } from "./planet/planet.component";

@Component({
  selector: 'app-about',
  imports: [PlanetComponent],
  templateUrl: './about.component.html',
  styleUrl: './about.component.css'
})
export class AboutComponent {

}
