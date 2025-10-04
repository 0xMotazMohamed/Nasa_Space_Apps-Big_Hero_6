import { Component, ViewChild, ElementRef, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ButtonModule } from "primeng/button";
import { RouterLink } from '@angular/router';

@Component({
  selector: 'app-hero',
  standalone: true,
  imports: [CommonModule, ButtonModule, RouterLink],
  templateUrl: './hero.component.html',
  styleUrls: ['./hero.component.css']
})
export class HeroComponent implements AfterViewInit {
  @ViewChild('starCanvas') starCanvas!: ElementRef<HTMLCanvasElement>;
  stars: { x: number; y: number; opacity: number; speedX: number; speedY: number }[] = [];

  constructor() {
    this.stars = Array.from({ length: 50 }, () => ({
      x: Math.random() * 100,
      y: Math.random() * 100,
      opacity: Math.random() * 0.7 + 0.3,
      speedX: (Math.random() - 0.1) * 0.1, // Random speed between -0.25 and 0.25
      speedY: (Math.random() - 0.1) * 0.1
    }));
  }

  ngAfterViewInit() {
    const canvas = this.starCanvas.nativeElement;
    const ctx = canvas.getContext('2d')!;
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      this.stars.forEach(star => {
        // Update position
        star.x += star.speedX;
        star.y += star.speedY;

        // Wrap around edges
        if (star.x < 0) star.x += 100;
        if (star.x > 100) star.x -= 100;
        if (star.y < 0) star.y += 100;
        if (star.y > 100) star.y -= 100;

        // Draw star
        ctx.beginPath();
        ctx.arc(star.x * canvas.width / 100, star.y * canvas.height / 100, 1.5, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 255, 255, ${star.opacity})`;
        ctx.fill();
      });
      requestAnimationFrame(animate);
    };
    animate();
  }
}
