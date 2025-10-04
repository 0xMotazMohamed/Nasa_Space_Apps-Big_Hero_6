import { Routes } from "@angular/router";
import { LandingLayoutComponent } from "./components/landing-layout/landing-layout.component";


export const LANDING_ROUTES: Routes = [
  {
    path: '',
    component: LandingLayoutComponent,
    children: [
      {
        path: '',
        pathMatch: 'full',
        redirectTo: 'home'
      },
      {
        path: 'home', loadComponent: () => import('./components/hero/hero.component').then(m => m.HeroComponent)
      },
      {
        path: 'about', loadComponent: () => import('./components/about/about.component').then(m => m.AboutComponent)
      },
      {
        path: 'team', loadComponent: () => import('./components/team/team.component').then(m => m.TeamComponent)
      }
    ]
  }
];
