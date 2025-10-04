import { Component, inject } from '@angular/core';
import { MapComponent } from "./solution/map.component";
import { ThemeService } from './services/theme.service';
import { ToastModule } from 'primeng/toast'
import { LoaderComponent } from "./shared/components/loader/loader.component";
import { AsyncPipe, NgIf } from '@angular/common';
import { LoaderService } from './services/loader.service';
import { Router, RouterOutlet } from '@angular/router';
import { AppearanceSettingsComponent } from "./shared/components/appearance-settings/appearance-settings.component";
@Component({
  selector: 'app-root',
  imports: [
    // MapComponent,
    ToastModule,
    LoaderComponent,
    NgIf,
    AsyncPipe,
    RouterOutlet,
    AppearanceSettingsComponent
],
templateUrl: './app.component.html',
styleUrl: './app.component.css',
})
export class AppComponent {
  title = 'weather-map';
  private theme = inject(ThemeService);
  public router = inject(Router)
  url:any = '';
  isLoading$:any;
  constructor(private loaderService: LoaderService) {}

  ngOnInit(){
    this.theme.init();
    this.isLoading$ = this.loaderService.isLoading$;
  }
}
