import { type Group, type Material, Mesh, MeshStandardMaterial, type Object3D, type Scene, Vector3 } from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

/*
 * Levels of detail. The pipeline writes, besides the full tiles, a much simpler model of every tile (pipeline/
 * boston_pipeline/lod.py): buildings as extruded footprints, the flat surfaces as a few large polygons, and the
 * largest trees as eight-triangle crowns. They come in files that each hold a block of tiles (a "super tile"), one
 * glTF node per tile named `cell_<tx>_<ty>`, so a block is fetched once and a tile's cell is shown or hidden alone.
 *
 * A cell is drawn only while the full tile with the same index is not: at the start, before the full tiles have
 * streamed in; beyond the full tiles' load radius; and for any tile dropped again when the walker moves on. A full
 * tile, once ready, hides its cell, so the two are never drawn together (no overlap, no z-fighting, no double
 * drawing). Cells are not solid, cannot be picked, are not described and cast and receive no shadow (the shadow map
 * covers only a few hundred metres); they only fill the view out to the fog.
 */

interface LodTiles {
  levels: {
    level: number;
    /** Base tiles along one side of a super tile. */
    super: number;
    tiles: { uri: string; super: [number, number]; cells: [number, number][] }[];
  }[];
}

interface LodFile {
  url: string;
  /** Scene position of the file's origin: the south-west corner of its super tile. */
  origin: Vector3;
  /** Scene position of the middle of the tiles the file has cells for. */
  centre: Vector3;
  cells: string[];
  state: 'idle' | 'loading' | 'ready' | 'failed';
  group?: Group;
  /** Each cell's node, by the key of the base tile it stands for (`<tx>_<ty>`). */
  nodes: Map<string, Object3D>;
}

/** Layers drawn with a shiny material, as the full tiles' water is. */
const WET = new Set(['lod_water']);

/** The simplified model of the whole city, in blocks that load as the walker gets near and unload as they leave. */
export class Lod {
  /** A file is kept loaded while its centre is within this many metres of the focus: for the pilot, everything. */
  radius = 9000;
  /** More files are not started while this many are on their way: they are small, and full tiles come first. */
  maxLoading = 2;

  private readonly files: LodFile[] = [];
  private readonly fileOfCell = new Map<string, LodFile>();
  /** Base tiles whose full version is loaded: their cells stay hidden. */
  private readonly covered = new Set<string>();
  private readonly materials = new Map<string, Material>();
  private readonly loader = new GLTFLoader();
  private loading = 0;
  /** Milliseconds the main thread has spent turning fetched files into scene objects (not counting the download). */
  private parseMs = 0;

  /**
   * @param tint Paint every cell pink, so a picture shows exactly where the simplified model is drawn and where the full tiles are
   * (`?lod=tint` in the address).
   */
  constructor(
    tileset: LodTiles,
    base: URL,
    tileSize: number,
    private readonly scene: Scene,
    private readonly tint = false,
  ) {
    for (const level of tileset.levels.filter((candidate) => candidate.level === 1)) {
      for (const entry of level.tiles) {
        const [i, j] = entry.super;
        const block = level.super * tileSize;
        let [minX, minY, maxX, maxY] = [Infinity, Infinity, -Infinity, -Infinity];
        for (const [tx, ty] of entry.cells) {
          minX = Math.min(minX, tx * tileSize);
          minY = Math.min(minY, ty * tileSize);
          maxX = Math.max(maxX, (tx + 1) * tileSize);
          maxY = Math.max(maxY, (ty + 1) * tileSize);
        }
        const file: LodFile = {
          url: new URL(entry.uri, base).href,
          origin: new Vector3(i * block, 0, -j * block),
          centre: new Vector3((minX + maxX) / 2, 0, -(minY + maxY) / 2),
          cells: entry.cells.map(([tx, ty]) => `${tx}_${ty}`),
          state: 'idle',
          nodes: new Map(),
        };
        this.files.push(file);
        for (const key of file.cells) this.fileOfCell.set(key, file);
      }
    }
  }

  /** Start fetching the blocks near `focus` (a scene position), nearest first, and drop the ones that are far. */
  update(focus: Vector3): void {
    const wanted: [number, LodFile][] = [];
    for (const file of this.files) {
      const distance = Math.hypot(file.centre.x - focus.x, file.centre.z - focus.z);
      if (file.state === 'idle' && distance < this.radius) wanted.push([distance, file]);
      else if (file.state === 'ready' && distance > this.radius + 1000) this.unload(file);
    }
    wanted.sort((a, b) => a[0] - b[0]);
    for (const [, file] of wanted) {
      if (this.loading >= this.maxLoading) break;
      void this.load(file);
    }
  }

  /** How many blocks within range of `focus` are still to arrive. */
  pending(focus: Vector3): number {
    let count = 0;
    for (const file of this.files) {
      if ((file.state === 'idle' || file.state === 'loading') && Math.hypot(file.centre.x - focus.x, file.centre.z - focus.z) < this.radius) count++;
    }
    return count;
  }

  /** The full tile `key` (`<tx>_<ty>`) has been loaded (hide its cell) or dropped (show it again). */
  setFull(key: string, loaded: boolean): void {
    if (loaded) this.covered.add(key);
    else this.covered.delete(key);
    const node = this.fileOfCell.get(key)?.nodes.get(key);
    if (node) node.visible = !loaded;
  }

  /** What is loaded and drawn, for checking: blocks, cells shown, the triangles of the cells shown, and the time spent building them. */
  stats(): { files: number; loaded: number; cellsShown: number; triangles: number; parseMs: number } {
    let loaded = 0;
    let cellsShown = 0;
    let triangles = 0;
    for (const file of this.files) {
      if (file.state !== 'ready') continue;
      loaded++;
      for (const node of file.nodes.values()) {
        if (!node.visible) continue;
        cellsShown++;
        node.traverse((child) => {
          if (child instanceof Mesh) triangles += (child.geometry.index?.count ?? 0) / 3;
        });
      }
    }
    return { files: this.files.length, loaded, cellsShown, triangles, parseMs: Math.round(this.parseMs) };
  }

  private async load(file: LodFile): Promise<void> {
    file.state = 'loading';
    this.loading++;
    try {
      // Fetched at low priority, so the full tiles beside them in the queue are served first.
      const response = await fetch(file.url, { priority: 'low' } as RequestInit);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const bytes = await response.arrayBuffer();
      const started = performance.now();
      const gltf = await this.loader.parseAsync(bytes, '');
      const group = gltf.scene;
      group.position.copy(file.origin);
      group.traverse((node) => {
        if (node instanceof Mesh) {
          node.material = this.materialFor(node.name, node.material as MeshStandardMaterial);
          node.castShadow = false;
          node.receiveShadow = false;
        }
      });
      for (const child of group.children) {
        const key = child.name.replace(/^cell_/, '');
        if (!this.fileOfCell.has(key)) continue;
        file.nodes.set(key, child);
        child.visible = !this.covered.has(key);
      }
      // Nothing here moves: compute every matrix once and never again.
      group.updateMatrixWorld(true);
      group.traverse((node) => (node.matrixAutoUpdate = false));
      file.group = group;
      file.state = 'ready';
      this.scene.add(group);
      this.parseMs += performance.now() - started;
    } catch (error) {
      console.error(`levels of detail ${file.url} failed to load`, error);
      file.state = 'failed';
    } finally {
      this.loading--;
    }
  }

  private unload(file: LodFile): void {
    file.group?.traverse((node) => {
      if (node instanceof Mesh) node.geometry.dispose();
    });
    if (file.group) this.scene.remove(file.group);
    file.group = undefined;
    file.nodes.clear();
    file.state = 'idle';
  }

  /** One shared material per layer: vertex colours, flat shading, no facade shader and no shadows. */
  private materialFor(layer: string, source: MeshStandardMaterial): Material {
    let material = this.materials.get(layer);
    if (!material) {
      material = new MeshStandardMaterial({
        color: this.tint ? '#ff4da6' : source.color, // multiplies the vertex colours
        roughness: source.roughness,
        metalness: WET.has(layer) ? 0.55 : source.metalness,
        vertexColors: source.vertexColors,
        flatShading: true, // the cells carry no normals: every face is shaded flat, as in the tiles
      });
      this.materials.set(layer, material);
    }
    source.dispose();
    return material;
  }
}
