import { Component } from '@angular/core';
import { CardModule } from 'primeng/card'
import { TagModule } from 'primeng/tag'
@Component({
  selector: 'app-team',
  imports: [CardModule, TagModule],
  templateUrl: './team.component.html',
  styleUrl: './team.component.css'
})
export class TeamComponent {
  team:any = [
    {
      img: 'adham.png',
      name: 'Adham M. Farouk',
      jobTitle: 'Frontend Developer',
    },
    {
      img: 'moataz.png',
      name: 'Moataz M. Ali',
      jobTitle: 'Data Science',
    },
    {
      img: 'hassan.png',
      name: 'Ahmed Hassan',
      jobTitle: 'Data Analyst',
    },
    {
      img: 'ahmed.png',
      name: 'Ahmed Abd El Moniem',
      jobTitle: 'JOKER (Data Science, Backend Developer)',
    },
    {
      img: 'mohanad.png',
      name: 'Mohanad Galal',
      jobTitle: 'Data Science',
    },
  ]
}
