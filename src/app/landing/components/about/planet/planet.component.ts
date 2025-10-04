import { NgIf } from '@angular/common';
import { Component, OnInit, ElementRef, ViewChild, AfterViewInit, HostListener } from '@angular/core';
import { animate, style, transition, trigger } from '@angular/animations';
import * as THREE from 'three';
import { OrbitControls } from '@three-ts/orbit-controls';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { ButtonModule } from 'primeng/button';

@Component({
  selector: 'app-planet',
  templateUrl: './planet.component.html',
  styleUrls: ['./planet.component.css'],
  imports: [ButtonModule, NgIf],
  animations: [
    trigger('slideInOut', [
      transition(':enter', [
        style({ transform: 'translateY(-100%)', opacity: 0 }),
        animate('500ms ease-in', style({ transform: 'translateY(0)', opacity: 0.6 })),
      ]),
      transition(':leave', [
        style({ transform: 'translateY(0)', opacity: 0.6 }),
        animate('500ms ease-out', style({ transform: 'translateY(-100%)', opacity: 0 })),
      ]),
    ]),
  ],
})
export class PlanetComponent implements OnInit, AfterViewInit {
  @ViewChild('rendererCanvas') private rendererCanvas!: ElementRef<HTMLCanvasElement>;

  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private renderer!: THREE.WebGLRenderer;
  private planet!: THREE.Mesh;
  private satellite!: THREE.Group;
  private controls!: OrbitControls;
  private cameraAnimationStartTime: number = 0;
  private cameraAnimationDuration: number = 0.8;
  private cameraInitialPosition: THREE.Vector3 = new THREE.Vector3(0, 500, 500);
  private cameraTargetPosition: THREE.Vector3 = new THREE.Vector3(0, 3, 3);
  private isCameraAnimating: boolean = true;
  isOverlayVisible: boolean = false;
  mobileWidth=0;
  ngOnInit(): void {
    this.updateMobileWidth();
  }
  @HostListener('window:resize')
  updateMobileWidth(): void {
    this.mobileWidth = window.innerWidth;
  }
  ngAfterViewInit(): void {
    this.createScene();
    this.createStarfield();
    this.createCamera();
    this.createRenderer();
    this.createPlanetAndSatellite();
    this.createLights();
    this.addControls();
    this.handleResize();
    this.cameraAnimationStartTime = Date.now() / 1000;
    this.animate();
  }

  toggleOverlay(): void {
    this.isOverlayVisible = !this.isOverlayVisible;
  }

  private createScene(): void {
    this.scene = new THREE.Scene();
    const appBgColor = '#1D2125';
    try {
      this.scene.background = new THREE.Color(appBgColor || '#000000');
    } catch (error) {
      console.warn('Invalid color for --app-bg, falling back to black', error);
      this.scene.background = new THREE.Color('#000000');
    }
  }

  private createStarfield(): void {
    const starCount = 5000;
    const starGeometry = new THREE.BufferGeometry();
    const starPositions = new Float32Array(starCount * 3);
    const starVelocities = new Float32Array(starCount * 3);
    const isMoving = new Uint8Array(starCount);
    const movePercentage = 0.2;

    for (let i = 0; i < starCount; i++) {
      starPositions[i * 3] = (Math.random() - 0.5) * 2000;
      starPositions[i * 3 + 1] = (Math.random() - 0.5) * 2000;
      starPositions[i * 3 + 2] = (Math.random() - 0.5) * 2000;

      isMoving[i] = Math.random() < movePercentage ? 1 : 0;

      if (isMoving[i]) {
        starVelocities[i * 3] = (Math.random() - 0.5) * 0.1;
        starVelocities[i * 3 + 1] = (Math.random() - 0.5) * 0.1;
        starVelocities[i * 3 + 2] = (Math.random() - 0.5) * 0.1;
      } else {
        starVelocities[i * 3] = 0;
        starVelocities[i * 3 + 1] = 0;
        starVelocities[i * 3 + 2] = 0;
      }
    }

    starGeometry.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
    starGeometry.setAttribute('velocity', new THREE.BufferAttribute(starVelocities, 3));
    starGeometry.setAttribute('isMoving', new THREE.BufferAttribute(isMoving, 1));

    const starMaterial = new THREE.PointsMaterial({
      color: 0xffffff,
      size: 2,
      sizeAttenuation: true,
    });

    const starfield = new THREE.Points(starGeometry, starMaterial);
    starfield.name = 'starfield'; // Assign a name for easy access
    this.scene.add(starfield);
  }

  private createCamera(): void {
    const aspectRatio = window.innerWidth / window.innerHeight;
    this.camera = new THREE.PerspectiveCamera(75, aspectRatio, 0.1, 1000);
    this.camera.position.copy(this.cameraInitialPosition);
    this.camera.lookAt(0, 0, 0);
  }

  private createRenderer(): void {
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.rendererCanvas.nativeElement,
      antialias: true,
    });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);
  }

  private createPlanetAndSatellite(): void {
    const geometry = new THREE.SphereGeometry(1, 32, 32);
    const textureLoader = new THREE.TextureLoader();
    textureLoader.load(
      'earth-texture.png',
      (texture) => {
        const material = new THREE.MeshStandardMaterial({ map: texture });
        this.planet = new THREE.Mesh(geometry, material);
        this.planet.position.set(0, 0, 0);
        this.scene.add(this.planet);

        const glowTexture = this.createGlowTexture();
        const glowMaterial = new THREE.SpriteMaterial({
          map: glowTexture,
          color: 0x87ceeb,
          transparent: true,
          blending: THREE.AdditiveBlending,
          opacity: 0.6,
        });
        const glowSprite = new THREE.Sprite(glowMaterial);
        glowSprite.scale.set(3, 3, 1);
        glowSprite.position.set(0, 0, 0);
        this.scene.add(glowSprite);
      },
      undefined,
      (error) => {
        const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
        this.planet = new THREE.Mesh(geometry, material);
        this.scene.add(this.planet);

        const glowTexture = this.createGlowTexture();
        const glowMaterial = new THREE.SpriteMaterial({
          map: glowTexture,
          color: 0x87ceeb,
          transparent: true,
          blending: THREE.AdditiveBlending,
          opacity: 0.5,
        });
        const glowSprite = new THREE.Sprite(glowMaterial);
        glowSprite.scale.set(3, 3, 1);
        glowSprite.position.set(0, 0, 0);
        this.scene.add(glowSprite);
      }
    );

    const loader = new GLTFLoader();
    textureLoader.load(
      'satelite-effect.png',
      (satelliteTexture) => {
        loader.load(
          'satelite.gltf',
          (gltf) => {
            this.satellite = gltf.scene;
            this.satellite.traverse((child) => {
              if (child instanceof THREE.Mesh) {
                child.material = new THREE.MeshStandardMaterial({ map: satelliteTexture });
              }
            });
            this.satellite.position.set(3, 0, 0);
            this.satellite.scale.set(0.07, 0.07, 0.07);
            this.satellite.rotateZ(0.5);
            this.scene.add(this.satellite);
          },
          (progress) => {},
          (error) => {}
        );
      },
      undefined,
      (error) => {
        loader.load(
          'satelite.gltf',
          (gltf) => {
            this.satellite = gltf.scene;
            this.satellite.traverse((child) => {
              if (child instanceof THREE.Mesh) {
                child.material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
              }
            });
            this.satellite.position.set(3, 0, 0);
            this.satellite.scale.set(0.07, 0.07, 0.07);
            this.satellite.rotation.y += 2;
            this.satellite.rotateZ(0.5);
            this.scene.add(this.satellite);
          },
          undefined,
          (error) => {}
        );
      }
    );
  }

  private createGlowTexture(): THREE.Texture {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const context = canvas.getContext('2d')!;
    const gradient = context.createRadialGradient(128, 128, 0, 128, 128, 128);
    gradient.addColorStop(0, 'white');
    gradient.addColorStop(0.1, 'white');
    gradient.addColorStop(0.5, 'rgba(255, 255, 255, 0.2)');
    gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
    context.fillStyle = gradient;
    context.fillRect(0, 0, 256, 256);
    const texture = new THREE.Texture(canvas);
    texture.needsUpdate = true;
    return texture;
  }

  private createLights(): void {
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.0);
    this.scene.add(ambientLight);

    const pointLight = new THREE.PointLight(0xffffff, 3);
    pointLight.position.set(5, 5, 5);
    this.scene.add(pointLight);
  }

  private addControls(): void {
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.5;
    this.controls.target = new THREE.Vector3(0, 0, 0);
    this.controls.update();
  }

  private handleResize(): void {
    window.addEventListener('resize', () => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      this.renderer.setSize(width, height);
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
    });
  }

  private animate(): void {
    try {
      requestAnimationFrame(() => this.animate());

      if (this.isCameraAnimating) {
        const currentTime = Date.now() / 1000;
        const elapsed = currentTime - this.cameraAnimationStartTime;
        const t = Math.min(elapsed / this.cameraAnimationDuration, 1);
        this.camera.position.lerpVectors(this.cameraInitialPosition, this.cameraTargetPosition, t);
        this.camera.lookAt(0, 0, 0);
        if (t >= 1) {
          this.isCameraAnimating = false;
        }
      }

      if (this.planet) {
        this.planet.rotation.y += 0.005;
      }

      if (this.satellite) {
        const time = Date.now() * -0.001;
        const orbitRadius = 3;
        this.satellite.position.x = orbitRadius * Math.cos(time * 0.5);
        this.satellite.position.z = orbitRadius * Math.sin(time * 0.5);
      }

      // Update moving stars
      const starfield = this.scene.getObjectByName('starfield') as THREE.Points;
      if (starfield) {
        const positionAttribute = starfield.geometry.getAttribute('position') as THREE.BufferAttribute;
        const velocityAttribute = starfield.geometry.getAttribute('velocity') as THREE.BufferAttribute;
        const isMovingAttribute = starfield.geometry.getAttribute('isMoving') as THREE.BufferAttribute;
        const positions = positionAttribute.array as Float32Array;
        const velocities = velocityAttribute.array as Float32Array;
        const isMoving = isMovingAttribute.array as Uint8Array;
        const boundary = 500; // Same as (2000 / 2) from createStarfield

        for (let i = 0; i < starfield.geometry.attributes?.['position'].count; i++) {
          if (isMoving[i]) {
            // Update position
            positions[i * 3] += velocities[i * 3]; // X
            positions[i * 3 + 1] += velocities[i * 3 + 1]; // Y
            positions[i * 3 + 2] += velocities[i * 3 + 2]; // Z

            // Wrap around if outside boundaries
            if (positions[i * 3] > boundary) positions[i * 3] -= 2 * boundary;
            else if (positions[i * 3] < -boundary) positions[i * 3] += 2 * boundary;
            if (positions[i * 3 + 1] > boundary) positions[i * 3 + 1] -= 2 * boundary;
            else if (positions[i * 3 + 1] < -boundary) positions[i * 3 + 1] += 2 * boundary;
            if (positions[i * 3 + 2] > boundary) positions[i * 3 + 2] -= 2 * boundary;
            else if (positions[i * 3 + 2] < -boundary) positions[i * 3 + 2] += 2 * boundary;
          }
        }

        // Mark position attribute as needing update
        positionAttribute.needsUpdate = true;
      }

      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    } catch (error) {}
  }
}
