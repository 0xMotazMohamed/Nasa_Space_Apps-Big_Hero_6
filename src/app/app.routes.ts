import { Routes } from '@angular/router';
import { MapComponent } from './solution/map.component';

export const routes: Routes = [
  {
    path: 'modules',
    title:'AERA - CAIRO', loadChildren: () => import('./landing/LANDING_ROUTES').then(m => m.LANDING_ROUTES),
  },
  {
    path: 'map',
    title:'AERA - CAIRO',
    component: MapComponent
  },
  {
    path: '**',
    redirectTo: 'modules',
  }
];
