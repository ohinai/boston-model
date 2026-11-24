import proj4, { type Converter } from 'proj4';

/** The world frame, as written by the pipeline into tileset.json. */
export interface WorldFrame {
  proj4: string;
  origin: [number, number];
  tile_size: number;
}

/**
 * Converts between world metres (x east, y north of the origin) and WGS84.
 *
 * The 3D scene uses three.js axes: scene x = east, scene y = up and
 * scene z = -north. Everything outside rendering talks in world metres.
 */
export class Geo {
  private readonly converter: Converter;

  constructor(private readonly frame: WorldFrame) {
    this.converter = proj4(frame.proj4);
  }

  toLonLat(x: number, y: number): [number, number] {
    const [lon, lat] = this.converter.inverse([x + this.frame.origin[0], y + this.frame.origin[1]]);
    return [lon, lat];
  }

  toLocal(lon: number, lat: number): [number, number] {
    const [east, north] = this.converter.forward([lon, lat]);
    return [east - this.frame.origin[0], north - this.frame.origin[1]];
  }
}
