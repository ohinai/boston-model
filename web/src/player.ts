import { Line3, MathUtils, type PerspectiveCamera, Vector3 } from 'three';
import type { City } from './city';

const RADIUS = 0.3;
const HEIGHT = 1.75;
const EYE_HEIGHT = 1.62;
const WALK_SPEED = 2.6; // metres per second; a brisk walk
const RUN_SPEED = 8;
const FLY_SPEED = 35;
const FLY_FAST_SPEED = 150;
const GRAVITY = -20;
const JUMP_SPEED = 5.2;
const MAX_FALL_SPEED = 40;
const TURN_RATE = 110; // degrees per second on the arrow keys
const LOOK_RATE = 0.11; // degrees per pixel of mouse travel
const SUBSTEPS = 5;
const WATER_LEVEL = -0.35; // standing lower than this means standing in water

/** True when a key press belongs to a form control rather than to the walker. */
export function isTyping(target: EventTarget | null): boolean {
  return (
    target instanceof HTMLTextAreaElement || target instanceof HTMLInputElement || target instanceof HTMLSelectElement
  );
}

/** Keyboard and mouse state. Mouse look works with pointer lock or by dragging. */
export class Input {
  private readonly keys = new Set<string>();
  private lookX = 0;
  private lookY = 0;
  private dragging = false;

  constructor(private readonly canvas: HTMLCanvasElement) {
    window.addEventListener('keydown', (event) => {
      if (isTyping(event.target)) return;
      this.keys.add(event.code);
      if (event.code === 'Space' || event.code.startsWith('Arrow')) event.preventDefault();
    });
    window.addEventListener('keyup', (event) => this.keys.delete(event.code));
    window.addEventListener('blur', () => this.keys.clear());
    canvas.addEventListener('mousedown', () => (this.dragging = true));
    window.addEventListener('mouseup', () => (this.dragging = false));
    document.addEventListener('mousemove', (event) => {
      if (this.locked || this.dragging) {
        this.lookX += event.movementX;
        this.lookY += event.movementY;
      }
    });
  }

  get locked(): boolean {
    return document.pointerLockElement === this.canvas;
  }

  down(...codes: string[]): boolean {
    return codes.some((code) => this.keys.has(code));
  }

  /** +1, 0 or -1 from a pair of opposing key groups. */
  axis(negative: string[], positive: string[]): number {
    return Number(this.down(...positive)) - Number(this.down(...negative));
  }

  /** Forget every held key, for when focus moves to a form. */
  release(): void {
    this.keys.clear();
    this.dragging = false;
  }

  takeLook(): [number, number] {
    const look: [number, number] = [this.lookX, this.lookY];
    this.lookX = this.lookY = 0;
    return look;
  }
}

/** A person-sized capsule that walks on the city, or flies through it. */
export class Player {
  /** Scene position of the feet. */
  readonly position = new Vector3();
  /** Degrees clockwise from north. */
  heading = 0;
  /** Degrees above the horizon. */
  pitch = 0;
  flying = false;
  /** Called when the walker is put back on land after stepping into water. */
  onRescue?: () => void;

  private readonly velocity = new Vector3();
  private onGround = false;
  private readonly lastDry = new Vector3();
  private hasDry = false;
  private readonly segment = new Line3();
  private readonly before = new Vector3();
  private readonly push = new Vector3();
  private readonly move = new Vector3();

  constructor(private readonly city: City) {}

  /** Place the player at a world-metre position. Altitude > 0 starts in the air. */
  teleport(x: number, y: number, heading: number, altitude = 0, pitch = 0): void {
    this.position.set(x, altitude, -y);
    this.heading = heading;
    this.pitch = pitch;
    this.velocity.set(0, 0, 0);
    this.onGround = false;
    this.hasDry = false;
  }

  update(dt: number, input: Input): void {
    const [lookX, lookY] = input.takeLook();
    this.heading += lookX * LOOK_RATE + input.axis(['ArrowLeft'], ['ArrowRight']) * TURN_RATE * dt;
    this.heading = MathUtils.euclideanModulo(this.heading, 360);
    this.pitch = MathUtils.clamp(this.pitch - lookY * LOOK_RATE, -89, 89);

    const forward = input.axis(['KeyS', 'ArrowDown'], ['KeyW', 'ArrowUp']);
    const strafe = input.axis(['KeyA'], ['KeyD']);
    const fast = input.down('ShiftLeft', 'ShiftRight');
    const yaw = MathUtils.degToRad(this.heading);
    const sin = Math.sin(yaw);
    const cos = Math.cos(yaw);

    if (this.flying) {
      const tilt = MathUtils.degToRad(this.pitch);
      const speed = (fast ? FLY_FAST_SPEED : FLY_SPEED) * dt;
      const climb = input.axis(['KeyC'], ['Space']);
      this.position.x += (sin * Math.cos(tilt) * forward + cos * strafe) * speed;
      this.position.z += (-cos * Math.cos(tilt) * forward + sin * strafe) * speed;
      this.position.y = Math.max(0.5, this.position.y + (Math.sin(tilt) * forward + climb) * speed);
      this.velocity.set(0, 0, 0);
      this.onGround = false;
      return;
    }

    // Hold still until there is ground to stand on.
    if (!this.city.groundReady(this.position)) return;

    this.move.set(sin * forward + cos * strafe, 0, -cos * forward + sin * strafe);
    if (this.move.lengthSq() > 0) this.move.normalize().multiplyScalar(fast ? RUN_SPEED : WALK_SPEED);
    if (this.onGround && input.down('Space')) {
      this.velocity.y = JUMP_SPEED;
      this.onGround = false;
    }
    for (let i = 0; i < SUBSTEPS; i++) this.step(dt / SUBSTEPS);

    if (this.onGround && this.position.y > WATER_LEVEL + 0.15) {
      this.lastDry.copy(this.position);
      this.hasDry = true;
    } else if (this.hasDry && ((this.onGround && this.position.y < WATER_LEVEL) || this.position.y < -30)) {
      this.position.copy(this.lastDry);
      this.velocity.set(0, 0, 0);
      this.onRescue?.();
    }
  }

  /** Put the camera at eye level, looking where the player looks. */
  applyTo(camera: PerspectiveCamera): void {
    camera.position.set(this.position.x, this.position.y + EYE_HEIGHT, this.position.z);
    camera.rotation.set(MathUtils.degToRad(this.pitch), -MathUtils.degToRad(this.heading), 0, 'YXZ');
  }

  private step(dt: number): void {
    const { position, velocity, segment, push } = this;
    velocity.y = this.onGround ? dt * GRAVITY : Math.max(-MAX_FALL_SPEED, velocity.y + dt * GRAVITY);
    position.addScaledVector(velocity, dt).addScaledVector(this.move, dt);

    // Stay inside the built area: there is nothing to stand on beyond it.
    const [minX, minY, maxX, maxY] = this.city.extent;
    position.x = MathUtils.clamp(position.x, minX + 1, maxX - 1);
    position.z = MathUtils.clamp(position.z, -maxY + 1, -minY - 1);

    segment.start.set(position.x, position.y + RADIUS, position.z);
    segment.end.set(position.x, position.y + HEIGHT - RADIUS, position.z);
    this.before.copy(segment.start);
    this.city.collide(segment, RADIUS);
    push.subVectors(segment.start, this.before);

    // Being pushed mostly upward means something is underfoot.
    this.onGround = push.y > Math.abs(dt * velocity.y * 0.25);
    const distance = push.length();
    if (distance > 1e-5) {
      push.divideScalar(distance);
      position.addScaledVector(push, distance - 1e-5);
      if (!this.onGround) velocity.addScaledVector(push, -push.dot(velocity));
    }
    if (this.onGround) velocity.set(0, 0, 0);
  }
}
