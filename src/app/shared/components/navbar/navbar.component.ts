import { ButtonModule } from 'primeng/button';
import { Component } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';

@Component({
  selector: 'app-navbar',
  imports: [RouterLink, RouterLinkActive, ButtonModule],
  templateUrl: './navbar.component.html',
  styleUrl: './navbar.component.css'
})
export class NavbarComponent {
  isNavbarExpanded = false;

  toggleNavbar(): void {
    this.isNavbarExpanded = !this.isNavbarExpanded;
  }
}
