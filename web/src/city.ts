import {
  Box3,
  BufferGeometry,
  Group,
  Line3,
  type Material,
  Mesh,
  MeshStandardMaterial,
  type Object3D,
  Raycaster,
  type Scene,
  Vector3,
} from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { type ExtendedTriangle, MeshBVH } from 'three-mesh-bvh';
import { facade } from './facade';
import type { WorldFrame } from './geo';
import { Trunks } from './trunks';

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
  id: string;
  name?: string;
  pos: [number, number];
}

/** A building as described in a tile's data file. */
export interface BuildingInfo extends Named {
  /** For a part of a larger building: that building's id. */
  part_of?: string;
  height: number;
  /** Where the height came from: a measured tag, a storey count, a guess, or a fix in the repository. */
  height_source: 'height' | 'levels' | 'default' | 'override';
}

/** Something drawn in a tile that is not a building: a bench, a statue, a bridge, a tree. */
export interface ObjectInfo extends Named {
  /** bench, lamp, waste_basket, drinking_water, flagpole, plaque, stele, obelisk, column, statue, sculpture, monument, fountain, bridge, tree, ... */
  kind: string;
  species?: string;
  height?: number;
  /** For a tree: the radius in metres its trunk is drawn with at 1.3 m. Absent when no trunk is drawn. */
  trunk?: number;
}

/** A named park, garden, playground, pitch or pond, clipped to one tile. */
interface NamedArea {
  id: string;
  name: string;
  kind: string;
  /** Size in square metres of the whole feature, not just the part in this tile. */
  area: number;
  /** Outer outlines of the part inside the tile, in world metres. */
  rings: [number, number][][];
}

/** What a ray from the eye hits first. */
export interface Pick {
  /** Scene position of the hit. */
  point: Vector3;
  distance: number;
  /** Tile layer that was hit, for example `green` or `building_walls`. */
  layer: string;
  building?: BuildingInfo;
  /** For furniture, monuments, bridges and trees: the object that was hit. */
  object?: ObjectInfo;
}

interface Semantics {
  streets: Street[];
  buildings: BuildingInfo[];
  pois: Named[];
  /** Absent in tiles built before objects were added. */
  objects?: ObjectInfo[];
  /** Absent in tiles built before named areas were added. */
  areas?: NamedArea[];
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
  /** The trees' trunks, which are solid although the `trees` layer is not (see trunks.ts). */
  trunks?: Trunks;
}

/** Layers a walker passes straight through: crowns, and a willow's hanging skirt. Trunks are solid (trunks.ts). */
const NOT_SOLID = new Set(['trees']);
const SMOOTH = new Set(['trees', 'figures']); // layers shaded smooth, not faceted: lumpy crowns, hand-made statues
const CASTS_SHADOW = new Set(['building_walls', 'building_roofs', 'trees', 'furniture', 'monuments', 'figures', 'bridges', 'barriers']);

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

/** Even-odd test: is the point inside the ring? */
function insideRing(x: number, y: number, ring: [number, number][]): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [ax, ay] = ring[i];
    const [bx, by] = ring[j];
    if (ay > y !== by > y && x < ((bx - ax) * (y - ay)) / (by - ay) + ax) inside = !inside;
  }
  return inside;
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
  private readonly raycaster = new Raycaster();
  private readonly tileOfGroup = new WeakMap<Object3D, Tile>();
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
    // Trunks first: whatever then pushes back (a wall, the ground) has the last word.
    for (const tile of this.tiles.values()) tile.trunks?.collide(segment, radius);
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

  /**
   * What a ray hits first, tested against the drawn geometry of every loaded tile.
   * Thorough rather than fast: meant for a single look, not for every frame.
   */
  pick(origin: Vector3, direction: Vector3, far = 600): Pick | null {
    this.raycaster.set(origin, direction);
    this.raycaster.far = far;
    const groups: Group[] = [];
    for (const tile of this.tiles.values()) if (tile.group) groups.push(tile.group);
    const hit = this.raycaster.intersectObjects(groups, true)[0];
    if (!hit) return null;
    const mesh = hit.object as Mesh;
    // Vertices carry an index into the tile's data file: buildings for building layers, objects for the rest.
    const ids = mesh.geometry.getAttribute('_feature_id_0');
    const semantics = mesh.parent ? this.tileOfGroup.get(mesh.parent)?.semantics : undefined;
    let building: BuildingInfo | undefined;
    let object: ObjectInfo | undefined;
    if (ids && hit.face && semantics) {
      const index = Math.round(ids.getX(hit.face.a));
      if (mesh.name === 'building_walls' || mesh.name === 'building_roofs') building = semantics.buildings?.[index];
      else object = semantics.objects?.[index];
    }
    return { point: hit.point.clone(), distance: hit.distance, layer: mesh.name, building, object };
  }

  /** The tile indices of a world-metre position. */
  tileIndex(x: number, y: number): [number, number] {
    const size = this.frame.tile_size;
    return [Math.floor(x / size), Math.floor(y / size)];
  }

  /** The street, the park (smallest named area) and the nearest named place at a world-metre position. */
  describe(x: number, y: number): { street?: string; park?: string; nearby?: string } {
    const size = this.frame.tile_size;
    const tx = Math.floor(x / size);
    const ty = Math.floor(y / size);
    let street: string | undefined;
    let park: string | undefined;
    let nearby: string | undefined;
    let parkArea = Infinity;
    // Areas are clipped per tile, so only the tile the point is in can contain it.
    for (const area of this.tiles.get(`${tx}_${ty}`)?.semantics?.areas ?? []) {
      if (area.area < parkArea && area.rings.some((ring) => insideRing(x, y, ring))) {
        parkArea = area.area;
        park = area.name;
      }
    }
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
    return { street, park, nearby };
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
        if (SMOOTH.has(layer)) node.geometry.computeVertexNormals(); // crowns are lumpy skins: smooth them, not faceted
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
      tile.trunks = Trunks.from(semantics?.objects, tile.origin);
      this.tileOfGroup.set(group, tile);
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
    tile.group = tile.bvh = tile.bounds = tile.semantics = tile.trunks = undefined;
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
        // Tiles carry no normals: every face is shaded flat, as the loader has already worked out.
        flatShading: SMOOTH.has(layer) ? false : source.flatShading, // trees and statues are smooth-shaded, everything else flat
      });
      if (layer === 'building_walls') facade(created);
      if (layer === 'water') created.metalness = 0.55;
      // Tree meshes ship smooth indexed normals; facet them.
      material = created;
      this.materials.set(layer, material);
    }
    source.dispose();
    return material;
  }
}
