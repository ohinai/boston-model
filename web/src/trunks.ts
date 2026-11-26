import type { Line3, Vector3 } from 'three';
import type { ObjectInfo } from './city';

// Tree trunks are solid; crowns and a willow's hanging skirt are not (the `trees` layer is left out of the
// tile's collision mesh, see NOT_SOLID in city.ts). A trunk is a vertical cylinder built from the tile's data
// file: the position of each tree and the radius its trunk is drawn with at 1.3 m, which the pipeline writes as
// `trunk`. It is tested directly, not put in the tile's BVH: a trunk is a circle in plan, a couple of hundred
// of them cost a few hundred distance tests, there is no mesh to build or to draw, and picking (which looks at
// drawn meshes only) never sees them.

/** Thinnest trunk worth stopping a walker (metres): a sapling's stem is thinner than the person who bumps it. */
const MIN_RADIUS = 0.15;
/** The collider starts this far under the ground the tile stands on, so a tree at a pond's edge or in a bed holds. */
const BELOW = 1.0;
/**
 * And reaches this far above it. The walker is 1.75 m tall and a jump rises 0.68 m, so from the ground nobody can
 * land on top of a trunk; the rest of the tree is foliage, which is not solid.
 */
const ABOVE = 3.0;
/** Closer than this (metres) to a trunk's axis there is no sideways direction to push in. */
const DEGENERATE = 1e-6;

/** The solid trunks of one tile, in scene coordinates. */
export class Trunks {
  /** How many trunks. */
  readonly count: number;
  /** Per trunk: scene x, scene z, radius. */
  private readonly data: Float32Array;
  private readonly floor: number;
  private readonly ceiling: number;
  /** Bounds of the trunks in plan, grown by the widest radius: a segment outside them cannot touch any. */
  private readonly minX: number;
  private readonly maxX: number;
  private readonly minZ: number;
  private readonly maxZ: number;

  /**
   * The trunks listed in a tile's data file, or undefined when it lists none (tiles built before trunks were
   * written have no `trunk`, and their trees stay as they were).
   * @param origin Scene position of the tile's own origin: the world metres in `pos` are absolute.
   */
  static from(objects: ObjectInfo[] | undefined, origin: Vector3): Trunks | undefined {
    const trunks = (objects ?? []).filter((o) => o.kind === 'tree' && typeof o.trunk === 'number' && o.trunk > 0);
    return trunks.length ? new Trunks(trunks, origin) : undefined;
  }

  private constructor(trees: ObjectInfo[], origin: Vector3) {
    this.count = trees.length;
    this.data = new Float32Array(trees.length * 3);
    this.floor = origin.y - BELOW;
    this.ceiling = origin.y + ABOVE;
    let minX = Infinity;
    let maxX = -Infinity;
    let minZ = Infinity;
    let maxZ = -Infinity;
    let widest = 0;
    trees.forEach((tree, index) => {
      const radius = Math.max(MIN_RADIUS, tree.trunk ?? 0);
      const x = tree.pos[0]; // world east, which is scene x
      const z = -tree.pos[1]; // world north, which is scene -z
      this.data.set([x, z, radius], index * 3);
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minZ = Math.min(minZ, z);
      maxZ = Math.max(maxZ, z);
      widest = Math.max(widest, radius);
    });
    this.minX = minX - widest;
    this.maxX = maxX + widest;
    this.minZ = minZ - widest;
    this.maxZ = maxZ + widest;
  }

  /**
   * Push a capsule (a scene-space segment with a radius) out of every trunk it overlaps, sideways. The segment
   * is moved in place, as in `City.collide`. Returns whether anything moved it.
   */
  collide(segment: Line3, radius: number): boolean {
    const { start, end } = segment;
    // Nothing to do when the capsule's box misses the trunks' box, in plan or in height.
    if (Math.max(start.x, end.x) + radius < this.minX || Math.min(start.x, end.x) - radius > this.maxX) return false;
    if (Math.max(start.z, end.z) + radius < this.minZ || Math.min(start.z, end.z) - radius > this.maxZ) return false;
    if (Math.max(start.y, end.y) + radius < this.floor || Math.min(start.y, end.y) - radius > this.ceiling) return false;

    let moved = false;
    const { data } = this;
    for (let i = 0; i < data.length; i += 3) {
      const cx = data[i];
      const cz = data[i + 1];
      const reach = radius + data[i + 2];
      // The point of the segment, seen from above, that is nearest the trunk's axis.
      const ax = start.x;
      const az = start.z;
      const dx = end.x - ax;
      const dz = end.z - az;
      const lengthSq = dx * dx + dz * dz;
      const t = lengthSq > 1e-12 ? Math.min(1, Math.max(0, ((cx - ax) * dx + (cz - az) * dz) / lengthSq)) : 0;
      const ox = ax + t * dx - cx;
      const oz = az + t * dz - cz;
      if (Math.abs(ox) >= reach || Math.abs(oz) >= reach) continue;
      const distance = Math.hypot(ox, oz);
      if (distance >= reach) continue;
      const push = reach - distance;
      // Dead centre has no direction to go: step aside from the segment (east if it is upright).
      let px = push;
      let pz = 0;
      if (distance > DEGENERATE) {
        px = (ox / distance) * push;
        pz = (oz / distance) * push;
      } else if (lengthSq > 1e-12) {
        const length = Math.sqrt(lengthSq);
        px = (-dz / length) * push;
        pz = (dx / length) * push;
      }
      start.x += px;
      start.z += pz;
      end.x += px;
      end.z += pz;
      moved = true;
    }
    return moved;
  }
}
