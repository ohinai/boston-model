import type { Object3D, PerspectiveCamera, Scene, Vector3, WebGLRenderer } from 'three';
import type { City } from './city';
import type { Geo } from './geo';
import type { Input, Player } from './player';

/** What capture mode needs from the rest of the viewer. */
export interface CaptureHost {
  renderer: WebGLRenderer;
  scene: Scene;
  camera: PerspectiveCamera;
  /** The sky dome, which follows the camera. */
  sky: Object3D;
  city: City;
  player: Player;
  geo: Geo;
  /** Size the canvas and camera to the picture: pixel ratio 1, aspect and vertical field of view in degrees. */
  resize(width: number, height: number, fov: number): void;
  /** Keep the shadow box on the ground around a position. */
  updateShadows(position: Vector3): void;
  /** Forget where the shadow box was, so the next update re-centres it. */
  resetShadows(): void;
}

interface CaptureView {
  id: string;
  lat: number;
  lon: number;
  heading: number;
  pitch: number;
}

interface CaptureJob {
  width: number;
  height: number;
  fov: number;
  post: string;
  views: CaptureView[];
}

interface Failure {
  id: string;
  reason: string;
}

const FIRST_TIMEOUT = 90_000; // milliseconds to wait for tiles at the first view
const LATER_TIMEOUT = 30_000;
const POLL = 100;
const SETTLE_STEPS = 90; // physics steps at 1/30 s: long enough to fall onto the ground
const JPEG_QUALITY = 0.9;
const DONE_URL = '/__capture/shot/done';

/** An input with nothing pressed and no mouse movement, for letting the walker settle. */
const IDLE_INPUT = {
  takeLook: (): [number, number] => [0, 0],
  axis: (): number => 0,
  down: (): boolean => false,
} as unknown as Input;

const sleep = (ms: number): Promise<void> => new Promise((resolve) => window.setTimeout(resolve, ms));
const message = (error: unknown): string => (error instanceof Error ? error.message : String(error));

/** POST a body, trying once more after a network error or a server error. */
async function post(url: string, body: BodyInit, type: string): Promise<void> {
  for (let attempt = 0; ; attempt++) {
    try {
      const response = await fetch(url, { method: 'POST', headers: { 'Content-Type': type }, body });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return;
    } catch (error) {
      if (attempt >= 1) throw error;
    }
  }
}

function validate(job: CaptureJob): void {
  const ok =
    job &&
    Number.isFinite(job.width) &&
    Number.isFinite(job.height) &&
    job.width > 0 &&
    job.height > 0 &&
    Number.isFinite(job.fov) &&
    typeof job.post === 'string' &&
    Array.isArray(job.views) &&
    job.views.every(
      (v) => typeof v.id === 'string' && Number.isFinite(v.lat) && Number.isFinite(v.lon),
    );
  if (!ok) throw new Error('The views file is malformed');
}

/**
 * Work through a list of views and send a picture of each to the server that asked.
 * Does not use requestAnimationFrame, which a headless or hidden page may never fire.
 */
export async function runCapture(host: CaptureHost, jobUrl: string): Promise<void> {
  document.body.classList.add('capture');
  let job: CaptureJob;
  try {
    const response = await fetch(jobUrl);
    if (!response.ok) throw new Error(`Could not load ${jobUrl} (HTTP ${response.status})`);
    job = (await response.json()) as CaptureJob;
    validate(job);
  } catch (error) {
    console.error(error);
    try {
      await post(DONE_URL, JSON.stringify({ captured: 0, failed: [], error: message(error) }), 'application/json');
    } catch (postError) {
      console.error(postError);
    }
    document.title = 'capture failed';
    return;
  }

  host.resize(job.width, job.height, job.fov);
  const failed: Failure[] = [];
  let captured = 0;
  for (const [index, view] of job.views.entries()) {
    try {
      await captureView(host, job, view, index === 0 ? FIRST_TIMEOUT : LATER_TIMEOUT);
      captured++;
    } catch (error) {
      console.error(`Capture of ${view.id} failed`, error);
      failed.push({ id: String(view.id), reason: message(error) });
    }
  }
  try {
    await post(`${job.post}done`, JSON.stringify({ captured, failed }), 'application/json');
  } catch (error) {
    console.error(error);
  }
  document.title = 'capture done';
}

async function captureView(host: CaptureHost, job: CaptureJob, view: CaptureView, timeout: number): Promise<void> {
  const { player, city, camera, renderer, scene, sky } = host;
  const [x, y] = host.geo.toLocal(view.lon, view.lat);
  player.flying = false;
  player.teleport(x, y, view.heading || 0, 0, view.pitch || 0);

  // Ask for tiles until everything wanted around the spot has arrived.
  const deadline = performance.now() + timeout;
  for (;;) {
    city.update(player.position);
    const { ready, wanted } = city.progress(player.position);
    if (ready >= wanted && city.groundReady(player.position)) break;
    if (performance.now() > deadline) throw new Error(`Timed out waiting for tiles (${ready} of ${wanted} ready)`);
    await sleep(POLL);
  }

  // Let the walker fall onto the ground with no keys pressed.
  for (let i = 0; i < SETTLE_STEPS; i++) player.update(1 / 30, IDLE_INPUT);
  player.applyTo(camera);
  camera.updateMatrixWorld();
  sky.position.copy(camera.position);
  host.resetShadows();
  host.updateShadows(player.position);

  // Render twice (the first may compile shaders), and read the picture in the same task.
  renderer.render(scene, camera);
  renderer.render(scene, camera);
  const blob = await new Promise<Blob | null>((resolve) =>
    renderer.domElement.toBlob(resolve, 'image/jpeg', JPEG_QUALITY),
  );
  if (!blob) throw new Error('Could not read the picture from the canvas');
  await post(`${job.post}${view.id}.jpg`, blob, 'image/jpeg');
}
