import './style.css';
import {
  ACESFilmicToneMapping,
  BackSide,
  CircleGeometry,
  Color,
  DirectionalLight,
  FogExp2,
  HemisphereLight,
  MathUtils,
  Mesh,
  MeshStandardMaterial,
  PerspectiveCamera,
  PMREMGenerator,
  Scene,
  ShaderMaterial,
  SphereGeometry,
  Vector3,
  WebGLRenderer,
} from 'three';
import { runCapture } from './capture';
import { City } from './city';
import { Geo } from './geo';
import { Notepad } from './notepad';
import { type NoteContext, buildCommit, siteUrl } from './notes';
import { Input, Player, isTyping } from './player';

const HORIZON = new Color('#c5d6e4');
const ZENITH = new Color('#3d76bf');
const SUN_AZIMUTH = 228; // degrees clockwise from north: a mid-afternoon sun
const SUN_ELEVATION = 38;
const SHADOW_RANGE = 280; // metres around a walker that receive shadows
const SHADOW_RANGE_AERIAL = 900; // and around the point below a flyer
const SUN_DISTANCE = 2000;

function element<T extends HTMLElement>(id: string): T {
  return document.getElementById(id) as T;
}

const canvas = element<HTMLCanvasElement>('view');
const renderer = new WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.9;
renderer.shadowMap.enabled = true;

const scene = new Scene();
scene.fog = new FogExp2(HORIZON, 0.00028);
const NEAR = 0.2; // metres: the closest thing drawn when on foot
const camera = new PerspectiveCamera(70, 1, NEAR, 16000);

/**
 * Push the near plane out as the camera climbs. Depth precision is spent between the
 * near and far planes, and from high up with a 20 cm near plane it runs out: surfaces
 * a few tens of centimetres apart, such as water and the land plane under it, flicker
 * in stripes. Nothing is ever closer than a few metres to a camera that high.
 */
function fitNearPlane(): void {
  const wanted = MathUtils.clamp(camera.position.y * 0.02, NEAR, 40);
  if (Math.abs(wanted - camera.near) > camera.near * 0.1) {
    camera.near = wanted;
    camera.updateProjectionMatrix();
  }
}

const azimuth = MathUtils.degToRad(SUN_AZIMUTH);
const elevation = MathUtils.degToRad(SUN_ELEVATION);
const sunDirection = new Vector3(
  Math.sin(azimuth) * Math.cos(elevation),
  Math.sin(elevation),
  -Math.cos(azimuth) * Math.cos(elevation),
);

function makeSky(): Mesh {
  const material = new ShaderMaterial({
    side: BackSide,
    depthWrite: false,
    fog: false,
    uniforms: { zenith: { value: ZENITH }, horizon: { value: HORIZON }, sun: { value: sunDirection } },
    vertexShader: /* glsl */ `
      varying vec3 vDirection;
      void main() {
        vDirection = position;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 zenith;
      uniform vec3 horizon;
      uniform vec3 sun;
      varying vec3 vDirection;
      void main() {
        vec3 direction = normalize(vDirection);
        vec3 colour = mix(horizon, zenith, pow(clamp(direction.y, 0.0, 1.0), 0.5));
        float towardSun = max(dot(direction, sun), 0.0);
        colour += vec3(1.0, 0.92, 0.78) * (pow(towardSun, 600.0) * 4.0 + pow(towardSun, 10.0) * 0.1);
        gl_FragColor = vec4(colour, 1.0);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`,
  });
  const sky = new Mesh(new SphereGeometry(1, 32, 16), material);
  sky.frustumCulled = false;
  return sky;
}

const sky = makeSky();
sky.scale.setScalar(14000);
sky.renderOrder = -1;
scene.add(sky);

// The same sky, small, rendered once into a reflection map for glass and water.
const reflectionScene = new Scene();
const reflectionSky = makeSky();
reflectionSky.scale.setScalar(50);
reflectionScene.add(reflectionSky);
scene.environment = new PMREMGenerator(renderer).fromScene(reflectionScene, 0, 0.1, 200).texture;
scene.environmentIntensity = 0.55;

scene.add(new HemisphereLight('#e2ebf8', '#9a9488', 1.5));
const sun = new DirectionalLight('#fff0dc', 3.0);
sun.castShadow = true;
sun.shadow.mapSize.set(4096, 4096);
// Shadow casters are drawn back-face-only, and nothing lit by the sun is a back
// face, so no depth bias is needed; any bias would leak light under walls.
sun.shadow.bias = 0;
sun.shadow.normalBias = 0.02;
scene.add(sun, sun.target);
const shadowCentre = new Vector3(Infinity, 0, 0);
const shadowFocus = new Vector3();
let shadowRange = 0;

/** Keep the shadow box on the ground around the player, wider when seen from the air. */
function updateShadows(position: Vector3): void {
  const range = position.y > 60 ? SHADOW_RANGE_AERIAL : SHADOW_RANGE;
  shadowFocus.set(position.x, 0, position.z);
  // Re-centre only now and then, so shadow edges do not swim.
  if (range === shadowRange && shadowCentre.distanceTo(shadowFocus) < range * 0.1) return;
  if (range !== shadowRange) {
    shadowRange = range;
    const lens = sun.shadow.camera;
    lens.left = lens.bottom = -range;
    lens.right = lens.top = range;
    lens.near = SUN_DISTANCE - range * 1.2 - 250;
    lens.far = SUN_DISTANCE + range * 1.2;
    lens.updateProjectionMatrix();
  }
  shadowCentre.copy(shadowFocus);
  sun.target.position.copy(shadowCentre);
  sun.position.copy(shadowCentre).addScaledVector(sunDirection, SUN_DISTANCE);
}

// Featureless land out to the horizon, just below the lowest built surface.
const beyond = new Mesh(new CircleGeometry(15000, 48), new MeshStandardMaterial({ color: '#8c9086', roughness: 1 }));
beyond.rotation.x = -Math.PI / 2;
beyond.position.y = -3; // well under the water surface, so the two never fight for the same pixels
scene.add(beyond);

// Capture mode (?capture=<url>) fixes the picture size itself.
const captureUrl = new URLSearchParams(window.location.search).get('capture');

function resize(): void {
  if (captureUrl) return;
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

let noticeTimer = 0;
function notice(text: string): void {
  const box = element('notice');
  box.textContent = text;
  box.hidden = false;
  window.clearTimeout(noticeTimer);
  noticeTimer = window.setTimeout(() => (box.hidden = true), 2600);
}

interface View {
  lat: number;
  lon: number;
  heading: number;
  altitude: number;
  pitch: number;
}

/** A shared location: #lat,lon,heading[,altitude,pitch]. */
function readHash(): View | null {
  const parts = window.location.hash.slice(1).split(',').map(Number);
  if (parts.length < 2 || !Number.isFinite(parts[0]) || !Number.isFinite(parts[1])) return null;
  const [lat, lon, heading = 0, altitude = 0, pitch = 0] = parts;
  return { lat, lon, heading: heading || 0, altitude: altitude || 0, pitch: pitch || 0 };
}

async function boot(): Promise<void> {
  const query = new URLSearchParams(window.location.search);
  const tilesetUrl = new URL(query.get('tileset') ?? 'tiles/tileset.json', document.baseURI);
  const city = await City.open(tilesetUrl, scene);
  const geo = new Geo(city.frame);
  const player = new Player(city);
  if (captureUrl) {
    await runCapture(
      {
        renderer,
        scene,
        camera,
        sky,
        city,
        player,
        geo,
        resize: (width, height, fov) => {
          renderer.setPixelRatio(1);
          renderer.setSize(width, height, false);
          camera.aspect = width / height;
          camera.fov = fov;
          camera.updateProjectionMatrix();
        },
        updateShadows,
        resetShadows: () => shadowCentre.set(Infinity, 0, 0),
      },
      captureUrl,
    );
    return;
  }
  const input = new Input(canvas);
  player.onRescue = () => notice('Splash. Back on dry land.');
  if (import.meta.env.DEV) Object.assign(window, { debug: { city, player, scene, renderer } });

  const modeButton = element<HTMLButtonElement>('mode');
  function setFlying(flying: boolean): void {
    player.flying = flying;
    modeButton.textContent = flying ? 'Fly' : 'Walk';
  }
  function go(view: View): void {
    const [x, y] = geo.toLocal(view.lon, view.lat);
    setFlying(view.altitude > 2);
    player.teleport(x, y, view.heading, view.altitude, view.pitch);
    shadowCentre.set(Infinity, 0, 0);
  }

  const places = element<HTMLSelectElement>('places');
  places.add(new Option('Jump to…', ''));
  city.places.forEach((place, index) => places.add(new Option(`${index + 1}  ${place.name}`, String(index))));
  function visit(index: number): void {
    const place = city.places[index];
    if (!place) return;
    go({ ...place, altitude: 0, pitch: 0 });
    notice(place.name);
  }
  places.addEventListener('change', () => {
    if (places.value) visit(Number(places.value));
    places.value = '';
    places.blur();
  });
  modeButton.addEventListener('click', () => {
    setFlying(!player.flying);
    modeButton.blur();
  });

  const start = city.places[0];
  go(readHash() ?? (start ? { ...start, altitude: 0, pitch: 0 } : { lat: 42.3587, lon: -71.063, heading: 0, altitude: 0, pitch: 0 }));

  const lookDirection = new Vector3();
  const round = (value: number, digits: number): number => Number(value.toFixed(digits));
  const notepad = new Notepad({
    scene,
    notice,
    // Everything a person or a tool needs to find the spot again and see what the author saw.
    capture: () => {
      const x = player.position.x;
      const y = -player.position.z;
      const z = player.position.y;
      const [lon, lat] = geo.toLonLat(x, y);
      const { street, park, nearby } = city.describe(x, y);
      player.applyTo(camera);
      camera.updateMatrixWorld();
      const hit = city.pick(camera.position, camera.getWorldDirection(lookDirection));
      let lookingAt: NoteContext['looking_at'];
      if (hit) {
        const [hitLon, hitLat] = geo.toLonLat(hit.point.x, -hit.point.z);
        lookingAt = {
          lat: round(hitLat, 7),
          lon: round(hitLon, 7),
          x: round(hit.point.x, 2),
          y: round(-hit.point.z, 2),
          z: round(hit.point.y, 2),
          distance: round(hit.distance, 2),
          layer: hit.layer,
          building: hit.building && {
            id: hit.building.id,
            part_of: hit.building.part_of,
            name: hit.building.name,
            height: hit.building.height,
            height_source: hit.building.height_source,
          },
          object: hit.object && {
            id: hit.object.id,
            kind: hit.object.kind,
            name: hit.object.name,
            species: hit.object.species,
          },
        };
      }
      const altitude = player.flying ? z.toFixed(0) : '0';
      const view = [lat.toFixed(6), lon.toFixed(6), player.heading.toFixed(0), altitude, player.pitch.toFixed(0)];
      return {
        standing: {
          lat: round(lat, 7),
          lon: round(lon, 7),
          x: round(x, 2),
          y: round(y, 2),
          z: round(z, 2),
          heading: round(player.heading, 1),
          pitch: round(player.pitch, 1),
          flying: player.flying,
        },
        looking_at: lookingAt,
        tile: city.tileIndex(x, y),
        street,
        park,
        near: nearby,
        link: `${siteUrl()}#${view.join(',')}`,
        data: city.sourceDate,
        build: buildCommit,
      };
    },
    go: ({ standing }) =>
      go({
        lat: standing.lat,
        lon: standing.lon,
        heading: standing.heading,
        altitude: standing.flying ? standing.z : 0,
        pitch: standing.pitch,
      }),
    releaseInput: () => {
      input.release();
      if (document.pointerLockElement) document.exitPointerLock();
    },
  });

  // Pasting or editing a location link moves the player without a reload.
  window.addEventListener('hashchange', () => {
    const view = readHash();
    if (view) go(view);
  });

  let started = false;
  const enter = element<HTMLButtonElement>('enter');
  const lockPointer = (): void => {
    // Not every embedding allows pointer lock; dragging and arrow keys still work.
    void Promise.resolve(canvas.requestPointerLock()).catch(() => undefined);
  };
  enter.addEventListener('click', () => {
    started = true;
    element('start').hidden = true;
    element('where').hidden = false;
    lockPointer();
  });
  canvas.addEventListener('click', () => {
    if (started && !input.locked) lockPointer();
  });
  document.addEventListener('pointerlockchange', () => (element('crosshair').hidden = !input.locked));
  window.addEventListener('keydown', (event) => {
    if (!started || event.repeat || isTyping(event.target)) return;
    if (event.code === 'KeyN') {
      event.preventDefault(); // or the N would be typed into the note
      notepad.open();
    } else if (event.code === 'KeyF') {
      setFlying(!player.flying);
      notice(player.flying ? 'Flying: Space rises, C sinks, Shift is fast' : 'Walking');
    } else if (/^Digit[1-9]$/.test(event.code)) {
      visit(Number(event.code.slice(5)) - 1);
    }
  });

  const date = city.sourceDate ? new Date(city.sourceDate).toLocaleDateString(undefined, { dateStyle: 'medium' }) : '';
  const credit = document.createElement('a');
  credit.href = 'https://www.openstreetmap.org/copyright';
  credit.target = '_blank';
  credit.rel = 'noopener';
  credit.textContent = 'OpenStreetMap contributors';
  element('attribution').append('Map data © ', credit, date ? `, ${date}` : '');

  let last = performance.now();
  let slowTick = 0;
  let hashTick = 0;
  function frame(now: number): void {
    const dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    if (started) player.update(dt, input);
    else input.takeLook();
    player.applyTo(camera);
    fitNearPlane();
    sky.position.copy(camera.position);

    updateShadows(player.position);

    if (now - slowTick > 200) {
      slowTick = now;
      city.update(player.position);
      const x = player.position.x;
      const y = -player.position.z;
      const [lon, lat] = geo.toLonLat(x, y);
      element('coords').textContent = `${lat.toFixed(5)}, ${lon.toFixed(5)}`;
      const { street, park, nearby } = city.describe(x, y);
      element('street').textContent = street ?? park ?? city.name;
      // The park is on the second line only when the first line is showing a street.
      element('nearby').textContent = [street ? park : undefined, nearby && `near ${nearby}`].filter(Boolean).join(' · ');
      notepad.update(player.position);

      if (!started) {
        const { ready, wanted } = city.progress(player.position);
        const walkable = city.groundReady(player.position) && ready > 0;
        element('area').textContent =
          ready < wanted ? `${city.name} · loading ${ready} of ${wanted} tiles` : `${city.name} · open data, walkable`;
        enter.disabled = !walkable;
        enter.textContent = walkable ? 'Start walking' : 'Loading…';
      }
      if (started && now - hashTick > 1000) {
        hashTick = now;
        let hash = `#${lat.toFixed(5)},${lon.toFixed(5)},${player.heading.toFixed(0)}`;
        if (player.flying) hash += `,${player.position.y.toFixed(0)},${player.pitch.toFixed(0)}`;
        if (hash !== window.location.hash) window.history.replaceState(null, '', hash);
      }
    }
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

boot().catch((error: unknown) => {
  console.error(error);
  element('area').textContent =
    `Could not load the city: ${error instanceof Error ? error.message : String(error)}. ` +
    'Build tiles first with the pipeline (see the README).';
});
