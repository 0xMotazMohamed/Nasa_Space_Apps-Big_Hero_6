import { provideHttpClient, withInterceptors } from '@angular/common/http';
import { ApplicationConfig, provideZoneChangeDetection } from '@angular/core';
import { MyPreset } from '../style';
import { provideAnimationsAsync } from '@angular/platform-browser/animations/async';
import { providePrimeNG } from 'primeng/config';
import { MessageService } from 'primeng/api';
import { DialogService } from 'primeng/dynamicdialog';
import { LoaderService } from './services/loader.service';
import { LoaderComponent } from './shared/components/loader/loader.component';
import { loaderInterceptor } from './interceptors/loader.interceptor';
import { routes } from './app.routes'
import { provideRouter } from '@angular/router';
export const appConfig: ApplicationConfig = {
  providers: [
    provideZoneChangeDetection({ eventCoalescing: true }),
    provideHttpClient(withInterceptors([loaderInterceptor])),
    provideRouter(routes),
    LoaderService,
    LoaderComponent,
    provideAnimationsAsync(),
        providePrimeNG({
          zIndex: {
            modal: 99999,   // dialog, sidebar
            overlay: 1000,  // dropdown, overlaypanel
            menu: 1000,     // overlay menus
            tooltip: 1100   // tooltip
          },
          theme: {
              preset: MyPreset,
              options: {
                darkModeSelector: '.app-dark'
              }
          }
        }),
        MessageService,
        DialogService
      ]
};
