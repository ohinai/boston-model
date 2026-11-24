import {
  Box3,
  BufferGeometry,
  Group,
  Line3,
  type Material,
  Mesh,
  MeshStandardMaterial,
  type Scene,
  Vector3,
} from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { type ExtendedTriangle, MeshBVH } from 'three-mesh-bvh';
import type { WorldFrame } from './geo';

export interface Place {
  name: string;
  lat: number;
  lon: number;
  heading: number;
  pos: [number, number];
}

interface Street {
  name: string;
  width: number;
  path: [number, number][];
}

interface Named {
  name?: string;
  pos: [number, number];
}

interface Semantics {
  streets: Street[];
  buildings: Named[];
  pois: Named[];
}

interface TilesetJson {
  root: {
    children: {
      transform: number[];
      content: { uri: string };
      extras: { tile: [number, number]; semantics: string };
    }[];
  };
  extras: {
    name: string;
    attribution: string;
    source_timestamp: string | null;
    world: WorldFrame;
    places: Place[];
  };
}

interface Tile {
  glb: string;
  semanticsUrl: string;
  /** Scene position of the tile's own origin (its south-west corner). */
  origin: Vector3;
  centre: Vector3;
  state: 'idle' | 'loading' | 'ready';
  group?: Group;
  bvh?: MeshBVH;
  /** Scene-space bounds of everything solid in the tile. */
  bounds?: Box3;
  semantics?: Semantics;
}

/** Layers a walker passes straight through. */
const NOT_SOLID = new Set(['trees']);
const CASTS_SHADOW = new Set(['building_walls', 'building_roofs', 'trees']);

// Building walls carry UVs counted in facade cells: one unit of u is a window
// bay, one unit of v a storey. This draws glazing into each cell and fades to
// a flat average tint once cells shrink below a few pixels.
const FACADE_GLSL = /* glsl */ `
  vec2 facadeCell = fract(vUv);
  bool streetLevel = vUv.y < 1.0;
  vec2 paneMin = streetLevel ? vec2(0.08, 0.06) : vec2(0.26, 0.3);
  vec2 paneMax = streetLevel ? vec2(0.92, 0.76) : vec2(0.74, 0.8);
  vec2 facadeBlur = fwidth(vUv) + 1e-5;
  vec2 pane = smoothstep(paneMin - facadeBlur, paneMin + facadeBlur, facadeCell)
    * (1.0 - smoothstep(paneMax - facadeBlur, paneMax + facadeBlur, facadeCell));
  float paneShare = (paneMax.x - paneMin.x) * (paneMax.y - paneMin.y);
  float facadeDetail = 1.0 - smoothstep(0.12, 0.45, max(facadeBlur.x, facadeBlur.y));
  // Walls too short for a bay have u == 0 everywhere and stay blank.
  float glazing = mix(paneShare, pane.x * pane.y, facadeDetail) * step(1e-6, fwidth(vUv.x));
  diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.1, 0.14, 0.19), glazing * 0.85);
`;

function facade(material: MeshStandardMaterial): void {
  material.defines = { ...material.defines, USE_UV: '' };
  material.onBeforeCompile = (shader) => {
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <color_fragment>', `#include <color_fragment>\n${FACADE_GLSL}`)
      .replace(
        '#include <roughnessmap_fragment>',
        '#include <roughnessmap_fragment>\nroughnessFactor = mix(roughnessFactor, 0.08, glazing);',
      )
      .replace(
        '#include <metalnessmap_fragment>',
        '#include <metalnessmap_fragment>\nmetalnessFactor = mix(metalnessFactor, 0.6, glazing);',
      );
  };
}

function distanceToPath(x: number, y: number, path: [number, number][]): number {
  let best = Infinity;
  for (let i = 1; i < path.length; i++) {
    const [ax, ay] = path[i - 1];
    const [bx, by] = path[i];
    const dx = bx - ax;
    const dy = by - ay;
    const lengthSq = dx * dx + dy * dy;
    const t = lengthSq ? Math.min(1, Math.max(0, ((x - ax) * dx + (y - ay) * dy) / lengthSq)) : 0;
    best = Math.min(best, Math.hypot(x - ax - t * dx, y - ay - t * dy));
  }
  return best;
}

/** The streamed city: loads tiles around a focus point and answers questions about them. */
export class City {
  readonly name: string;
  readonly attribution: string;
  readonly sourceDate: string | null;
  readonly frame: WorldFrame;
  readonly places: Place[];
  /** World-metre rectangle covered by tiles: [min x, min y, max x, max y]. */
  readonly extent: [number, number, number, number];
  /** Tiles whose centre is within this many metres of the focus are kept loaded. */
  loadRadius = 1800;

  private readonly tiles = new Map<string, Tile>();
  private readonly materials = new Map<string, Material>();
  private readonly loader = new GLTFLoader();
  private loading = 0;
  private readonly box = new Box3();
  private readonly trianglePoint = new Vector3();
  private readonly capsulePoint = new Vector3();

  static async open(url: URL, scene: Scene): Promise<City> {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Could not load ${url} (HTTP ${response.status})`);
    return new City((await response.json()) as TilesetJson, url, scene);
  }

  private constructor(tileset: TilesetJson, base: URL, private readonly scene: Scene) {
    const { extras } = tileset;
    this.name = extras.name;
    this.attribution = extras.attribution;
    this.sourceDate = extras.source_timestamp;
    this.frame = extras.world;
    this.places = extras.places ?? [];
    const size = this.frame.tile_size;
    const extent: [number, number, number, number] = [Infinity, Infinity, -Infinity, -Infinity];
    for (const child of tileset.root.children) {
      const [tx, ty] = child.extras.tile;
      // 3D Tiles transforms are column-major; elements 12..14 are the translation (z up).
      const [x, y, z] = child.transform.slice(12, 15);
      this.tiles.set(`${tx}_${ty}`, {
        glb: new URL(child.content.uri, base).href,
        semanticsUrl: new URL(child.extras.semantics, base).href,
        origin: new Vector3(x, z, -y),
        centre: new Vector3(x + size / 2, z, -(y + size / 2)),
        state: 'idle',
      });
      extent[0] = Math.min(extent[0], x);
      extent[1] = Math.min(extent[1], y);
      extent[2] = Math.max(extent[2], x + size);
      extent[3] = Math.max(extent[3], y + size);
    }
    this.extent = extent;
  }

  /** Start loading what is near `focus` (a scene position) and drop what is far. */
  update(focus: Vector3): void {
    const wanted: [number, Tile][] = [];
    for (const tile of this.tiles.values()) {
      const distance = Math.hypot(tile.centre.x - focus.x, tile.centre.z - focus.z);
      if (tile.state === 'idle' && distance < this.loadRadius) wanted.push([distance, tile]);
      else if (tile.state === 'ready' && distance > this.loadRadius + 400) this.unload(tile);
    }
    wanted.sort((a, b) => a[0] - b[0]);
    for (const [, tile] of wanted) {
      if (this.loading >= 6) break;
      void this.load(tile);
    }
  }

  /** How many tiles in range of `focus` are loaded, out of how many should be. */
  progress(focus: Vector3): { ready: number; wanted: number } {
    let ready = 0;
    let wanted = 0;
    for (const tile of this.tiles.values()) {
      if (Math.hypot(tile.centre.x - focus.x, tile.centre.z - focus.z) >= this.loadRadius) continue;
      wanted++;
      if (tile.state === 'ready') ready++;
    }
    return { ready, wanted };
  }

  /** False while the tile under a scene position is still on its way. */
  groundReady(position: Vector3): boolean {
    const tile = this.tileAt(position.x, -position.z);
    return !tile || tile.state === 'ready';
  }

  /**
   * Push a capsule (a scene-space segment with a radius) out of anything solid.
   * The segment is moved in place.
   */
  collide(segment: Line3, radius: number): void {
    const { box, trianglePoint, capsulePoint } = this;
    for (const tile of this.tiles.values()) {
      if (!tile.bvh || !tile.bounds) continue;
      box.makeEmpty().expandByPoint(segment.start).expandByPoint(segment.end).expandByScalar(radius);
      if (!box.intersectsBox(tile.bounds)) continue;
      // The BVH lives in tile space, which differs from scene space by a translation only.
      segment.start.sub(tile.origin);
      segment.end.sub(tile.origin);
      box.min.sub(tile.origin);
      box.max.sub(tile.origin);
      tile.bvh.shapecast({
        intersectsBounds: (bounds) => bounds.intersectsBox(box),
        intersectsTriangle: (triangle: ExtendedTriangle) => {
          const distance = triangle.closestPointToSegment(segment, trianglePoint, capsulePoint);
          if (distance < radius) {
            const push = capsulePoint.sub(trianglePoint).normalize().multiplyScalar(radius - distance);
            segment.start.add(push);
            segment.end.add(push);
          }
          return false;
        },
      });
      segment.start.add(tile.origin);
      segment.end.add(tile.origin);
    }
  }

  /** The street and the nearest named place at a world-metre position. */
  describe(x: number, y: number): { street?: string; nearby?: string } {
    const size = this.frame.tile_size;
    const tx = Math.floor(x / size);
    const ty = Math.floor(y / size);
    let street: string | undefined;
    let nearby: string | undefined;
    let streetDistance = 30;
    let nearbyDistance = 40;
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        const semantics = this.tiles.get(`${tx + dx}_${ty + dy}`)?.semantics;
        if (!semantics) continue;
        for (const candidate of semantics.streets) {
          const distance = distanceToPath(x, y, candidate.path) - candidate.width / 2;
          if (distance < streetDistance) {
            streetDistance = distance;
            street = candidate.name;
          }
        }
        for (const candidate of [...semantics.pois, ...semantics.buildings]) {
          if (!candidate.name) continue;
          const distance = Math.hypot(candidate.pos[0] - x, candidate.pos[1] - y);
          if (distance < nearbyDistance) {
            nearbyDistance = distance;
            nearby = candidate.name;
          }
        }
      }
    }
    return { street, nearby };
  }

  private tileAt(x: number, y: number): Tile | undefined {
    const size = this.frame.tile_size;
    return this.tiles.get(`${Math.floor(x / size)}_${Math.floor(y / size)}`);
  }

  private async load(tile: Tile): Promise<void> {
    tile.state = 'loading';
    this.loading++;
    try {
      const [gltf, semantics] = await Promise.all([
        this.loader.loadAsync(tile.glb),
        fetch(tile.semanticsUrl).then((r) => (r.ok ? (r.json() as Promise<Semantics>) : undefined)),
      ]);
      const group = gltf.scene;
      group.position.copy(tile.origin);
      const solid: BufferGeometry[] = [];
      group.traverse((node) => {
        if (!(node instanceof Mesh)) return;
        const layer = node.name;
        node.material = this.materialFor(layer, node.material as MeshStandardMaterial);
        node.castShadow = CASTS_SHADOW.has(layer);
        node.receiveShadow = true;
        if (!NOT_SOLID.has(layer)) {
          const geometry = new BufferGeometry();
          geometry.setAttribute('position', node.geometry.getAttribute('position'));
          geometry.setIndex(node.geometry.getIndex());
          solid.push(geometry);
        }
      });
      const merged = solid.length ? mergeGeometries(solid) : null;
      if (merged) {
        tile.bvh = new MeshBVH(merged);
        merged.computeBoundingBox();
        tile.bounds = merged.boundingBox!.clone().translate(tile.origin);
      }
      group.updateMatrixWorld(true);
      group.matrixAutoUpdate = false;
      tile.group = group;
      tile.semantics = semantics;
      tile.state = 'ready';
      this.scene.add(group);
    } catch (error) {
      console.error(`tile ${tile.glb} failed to load`, error);
      tile.state = 'idle';
    } finally {
      this.loading--;
    }
  }

  private unload(tile: Tile): void {
    tile.group?.traverse((node) => {
      if (node instanceof Mesh) node.geometry.dispose();
    });
    if (tile.group) this.scene.remove(tile.group);
    tile.group = tile.bvh = tile.bounds = tile.semantics = undefined;
    tile.state = 'idle';
  }

  /** One shared material per layer, taking its colour and finish from the tile data. */
  private materialFor(layer: string, source: MeshStandardMaterial): Material {
    let material = this.materials.get(layer);
    if (!material) {
      const created = new MeshStandardMaterial({
        color: source.color,
        roughness: source.roughness,
        metalness: source.metalness,
        vertexColors: source.vertexColors,
      });
      if (layer === 'building_walls') facade(created);
      if (layer === 'water') created.metalness = 0.55;
      material = created;
      this.materials.set(layer, material);
    }
    source.dispose();
    return material;
  }
}
