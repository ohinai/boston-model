import type { MeshStandardMaterial } from 'three';

/*
 * Building facades, drawn by one fragment shader. Walls carry no textures: their UVs count facade
 * cells (one bay by one storey, see pipeline/boston_pipeline/mesh.py), and every window, brick and
 * cornice here is computed from the cell coordinates, so it costs no extra triangles or downloads.
 *
 * What the shader reads from a wall:
 *   vUv.x  = (per-wall seed, 0 to 3, plus 4 if the wall starts above the ground, plus 8 times the wall's style) * 256
 *            + the bay number along the wall. Style 1 is a ballpark's outside wall (pipeline/boston_pipeline/stadium.py).
 *   vUv.y  = (number of storeys) * 64 + the storey number up the wall, so the top of the wall is known
 *            (for the cornice). Both offsets are whole numbers; the cell pattern repeats once per unit
 *            and does not notice them. mesh.py (`add_walls(facade=True)`) writes them.
 *   colour = the wall colour (a palette guess, or a building:colour tag). That decides the kind of
 *            facade: reddish is brick, cool grey-blue on a tall building is a glass curtain wall,
 *            other tall buildings get ribbon windows between precast panels, the rest is cut stone.
 * A wall too short for a bay has u == 0 and stays blank (but a curved front made of many short facets
 * has u running on from one facet to the next, so its windows carry round the curve).
 *
 * Nothing flickers at a distance because every pattern is filtered by its own size on screen:
 *   - rectangles (window openings, trim, cornice) are integrated over the pixel's footprint
 *     (fCover), so an edge is smooth and a feature narrower than a pixel keeps its share of the
 *     colour instead of aliasing;
 *   - repeating patterns (brick courses, block joints) fade to their average colour once a course
 *     is under about two pixels, separately along and across the wall;
 *   - the whole cell (windows against wall) fades to its average tint once a cell is a few pixels
 *     across (`detail`), and the fine structure inside a window (sash bars, the recess) fades first
 *     (`fine`).
 */

const FACADE_FUNCTIONS = /* glsl */ `
  // The sky in web/src/main.ts (HORIZON, ZENITH) in linear light, for glass to reflect.
  const vec3 SKY_HORIZON = vec3(0.558, 0.672, 0.775);
  const vec3 SKY_ZENITH = vec3(0.047, 0.181, 0.521);
  const float BAY = 3.0;     // metres per u (mesh.py BAY_WIDTH)
  const float STOREY = 3.5;  // metres per v (mesh.py FLOOR_HEIGHT)

  vec3 fEmit;       // light the facade sends out: the sky reflected in glass
  float fGlass;     // how much of this pixel is glass (smoother, so it glints)
  float fVary;      // how much the random differences between panes and windows show: none once they are a few pixels across
  bool fRaised;     // the wall starts above the ground, on a podium: no shop front, no base course
  vec3 fView;       // the view direction in the wall's axes: along the wall, up, out of the wall
  float fSky;       // how far above the horizon the view reflects (-1 to 1)

  float fHash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
  }

  // The share of the pixel interval [x - w/2, x + w/2] that lies inside [lo, hi].
  float fCover(float x, float lo, float hi, float w) {
    w = max(w, 1e-5);
    return clamp(min(hi, x + 0.5 * w) - max(lo, x - 0.5 * w), 0.0, w) / w;
  }
  float fBox(vec2 p, vec2 lo, vec2 hi, vec2 w) {
    return fCover(p.x, lo.x, hi.x, w.x) * fCover(p.y, lo.y, hi.y, w.y);
  }
  // The same for an interval repeated every unit.
  float fBand(float x, float lo, float hi, float w) {
    return fCover(x, lo, hi, w) + fCover(x, lo + 1.0, hi + 1.0, w) + fCover(x, lo - 1.0, hi - 1.0, w);
  }

  vec3 fSkyColour(float y) {
    vec3 sky = mix(SKY_HORIZON, SKY_ZENITH, pow(clamp(y, 0.0, 1.0), 0.5));
    vec3 street = vec3(0.14, 0.16, 0.18); // below the horizon, glass shows the street and the buildings opposite
    return mix(street, sky, smoothstep(-0.1, 0.3, y));
  }

  // What glass reflects: more at a grazing angle, a little different from pane to pane.
  vec3 fReflection(float f0, vec2 id, vec3 tint) {
    float fresnel = mix(f0, 0.85, pow(1.0 - clamp(fView.z, 0.0, 1.0), 4.0));
    return fSkyColour(fSky) * fresnel * mix(1.0, mix(0.75, 1.25, fHash(id + 5.5)), fVary) * tint;
  }

  // The room behind a pane: mostly dark, now and then a lit room or a drawn blind.
  vec3 fInterior(vec2 id, float lit) {
    float roll = fHash(id + 9.2);
    vec3 dark = vec3(0.016, 0.022, 0.032) * mix(0.6, 2.2, fHash(id + 3.7));
    vec3 room = vec3(0.24, 0.16, 0.08) * mix(0.5, 1.0, fHash(id + 1.3));
    vec3 blind = vec3(0.17, 0.16, 0.13);
    float roomOdds = mix(0.04, 0.28, lit);       // lit: 0 an ordinary window, 1 a shop, below 0 fewer lit rooms
    float blindOdds = roomOdds + mix(0.10, 0.02, lit);
    vec3 picked = roll < roomOdds ? room : (roll < blindOdds ? blind : dark);
    return mix(vec3(0.022, 0.03, 0.044), picked, fVary);
  }

  // A recessed, glazed opening in a wall. lo and hi bound it in cell fractions. sash: 0 two over
  // two, 1 six over six, 2 one pane, 3 a shop window. 'detail' and 'fine' say how much of the
  // structure to draw (see the top of this file).
  vec3 fWindow(vec3 wall, vec3 jamb, vec3 frame, vec2 f, vec2 id, vec2 fw, vec2 lo, vec2 hi,
               float depth, float sash, float fine, float f0, float lit) {
    float opening = fBox(f, lo, hi, fw);
    if (opening < 0.003) return wall;
    // Where the sight line through this point meets the glass, set back by 'depth' metres: it moves with the viewing angle.
    vec2 off = -fView.xy / max(fView.z, 0.25) * depth / vec2(BAY, STOREY);
    vec2 g = f + clamp(off, vec2(-0.25), vec2(0.25)) * fine;
    float seen = min(fBox(g, lo, hi, fw), opening);
    float onGlass = seen / opening;
    // A sight line that misses the glass meets the soffit (dark), a jamb or the sill (lit).
    vec3 side = jamb * (g.y > hi.y ? 0.3 : (g.y < lo.y ? 0.9 : 0.55));
    vec2 size = hi - lo;
    vec2 q = (g - lo) / size;
    vec2 qw = fw / size;
    float bars = 1.0 - fBox(q, vec2(0.06, 0.04), vec2(0.94, 0.96), qw);   // the frame all round
    if (sash < 1.5) {
      bars += fBox(q, vec2(0.0, 0.47), vec2(1.0, 0.53), qw);               // the meeting rail between the sashes
      bars += fBox(q, vec2(0.488, 0.0), vec2(0.512, 1.0), qw);             // the centre muntin
      if (sash > 0.5) {
        bars += fBox(q, vec2(0.0, 0.150), vec2(1.0, 0.168), qw) + fBox(q, vec2(0.0, 0.305), vec2(1.0, 0.323), qw);
        bars += fBox(q, vec2(0.0, 0.677), vec2(1.0, 0.695), qw) + fBox(q, vec2(0.0, 0.832), vec2(1.0, 0.850), qw);
      }
    } else if (sash > 2.5) {
      bars += fBox(q, vec2(0.49, 0.0), vec2(0.51, 1.0), qw) + fBox(q, vec2(0.0, 0.80), vec2(1.0, 0.84), qw);
    }
    bars = mix(0.12, clamp(bars, 0.0, 1.0), fine);
    vec3 pane = fInterior(id, lit);
    float glass = opening * onGlass * (1.0 - bars);
    fGlass = glass;
    fEmit = fReflection(f0, id, vec3(1.0)) * glass;
    vec3 inside = mix(side, mix(pane, frame, bars), onGlass);
    return mix(wall, inside, opening);
  }

  // Mortar between bricks: lighter and greyer than the brick.
  vec3 fMortar(vec3 base) {
    return mix(base * 1.3, vec3(0.36, 0.34, 0.31), 0.65);
  }

  // The wall itself: courses of brick, or blocks of stone (with a heavier bed joint on a ground floor of stone).
  vec3 fCourses(vec3 base, vec2 f, vec2 id, vec2 fw, float seed, bool brick, bool ground) {
    float courses = brick ? 46.0 : 9.0;
    float perBay = brick ? 15.0 : 3.0;
    float bed = brick ? 0.16 : (ground ? 0.16 : 0.07);
    float head = brick ? 0.07 : 0.025;
    float cy = f.y * courses;
    float rowIndex = floor(cy);
    float cx = f.x * perBay + 0.5 * mod(rowIndex, 2.0);
    vec2 blockId = vec2(floor(cx), rowIndex) + id * vec2(perBay, courses);
    vec2 bf = vec2(fract(cx), fract(cy));
    vec2 bw = fw * vec2(perBay, courses);
    float visY = 1.0 - smoothstep(0.35, 0.75, bw.y);
    float visX = visY * (1.0 - smoothstep(0.35, 0.75, bw.x));
    float jy = fBand(bf.y, 0.0, bed, bw.y);
    float jx = fBand(bf.x, 0.0, head, bw.x);
    float joints = mix(bed + head - bed * head, jy + (1.0 - jy) * mix(head, jx, visX), visY);
    float hA = fHash(blockId + seed * 7.0);
    float hB = fHash(blockId * 1.37 + 3.1);
    vec3 tone = brick
      ? vec3(mix(0.78, 1.22, hA)) * vec3(1.0, mix(0.93, 1.07, hB), mix(0.90, 1.10, hB))
      : vec3(mix(0.93, 1.07, hA));
    if (brick && hB < 0.04) tone *= 0.6; // the odd dark, hard-fired brick
    tone = mix(vec3(1.0), tone, visX);
    vec3 block = base * (brick ? 0.84 : 1.0) * tone;
    vec3 mortar = brick ? fMortar(base) : base * 0.78;
    vec3 wall = mix(block, mortar, joints);
    // A cell's tone drifts a little from one bay to the next.
    wall *= mix(1.0, mix(0.95, 1.05, fHash(id + seed)), fVary);
    return wall;
  }

  // Walls of brick or cut stone with punched windows.
  vec3 fPunched(vec3 base, vec2 f, vec2 id, vec2 fw, float floors, float v, float topDist,
                float seed, bool brick, float detail, float fine) {
    bool ground = id.y < 0.5 && !fRaised;
    bool topRow = floors > 0.5 && id.y > floors - 1.5;
    float seed2 = fHash(vec2(seed * 13.7, 2.1));
    float seed3 = fHash(vec2(seed * 5.3, 8.7));

    vec3 wall = fCourses(base, f, id, fw, seed, brick, ground);

    // Trim: pale limestone round brick, a moulded surround on stone.
    vec3 trim = brick ? vec3(0.46, 0.44, 0.40) : base * 1.14;
    float trimVis = 1.0 - smoothstep(0.08, 0.22, max(fw.x, fw.y));

    vec2 lo = vec2(0.31, ground ? 0.14 : 0.24);
    vec2 hi = vec2(0.69, 0.80);
    float sash = seed2 < 0.45 ? 0.0 : (seed2 < 0.85 ? 1.0 : 2.0);
    float lit = 0.0;
    vec3 frame = seed3 < 0.7 ? vec3(0.50, 0.49, 0.46) : vec3(0.05, 0.055, 0.05);
    if (topRow) hi.y = 0.74;
    bool shop = brick && ground && seed > 0.42;
    float fascia = 0.0;
    vec3 fasciaColour = vec3(0.0);
    if (shop) {
      lo = vec2(0.07, 0.13);
      hi = vec2(0.93, 0.70);
      sash = 3.0;
      lit = 1.0;
      frame = seed3 < 0.5 ? vec3(0.04, 0.04, 0.045) : vec3(0.45, 0.44, 0.40);
      fascia = fBox(f, vec2(0.0, 0.72), vec2(1.0, 0.95), fw) * step(seed2, 0.72);
      float pick = fHash(vec2(seed * 3.9, 4.4));
      fasciaColour = pick < 0.25 ? vec3(0.02, 0.075, 0.04)
                   : pick < 0.5 ? vec3(0.015, 0.03, 0.10)
                   : pick < 0.72 ? vec3(0.035, 0.035, 0.04)
                   : pick < 0.88 ? vec3(0.12, 0.025, 0.025) : vec3(0.5, 0.45, 0.35);
      wall = mix(wall, fasciaColour, fascia * trimVis);
    } else {
      float lintel = fBox(f, vec2(lo.x - 0.05, hi.y), vec2(hi.x + 0.05, hi.y + 0.055), fw);
      float sill = fBox(f, vec2(lo.x - 0.045, lo.y - 0.04), vec2(hi.x + 0.045, lo.y), fw);
      float shadow = fBox(f, vec2(lo.x - 0.03, lo.y - 0.10), vec2(hi.x + 0.03, lo.y - 0.04), fw);
      wall *= 1.0 - 0.28 * shadow * trimVis;
      wall = mix(wall, trim, clamp(lintel + sill, 0.0, 1.0) * trimVis);
      if (!brick) {
        float around = fBox(f, lo - vec2(0.045, 0.035), hi + vec2(0.045, 0.0), fw);
        wall = mix(wall, trim, around * trimVis);
      }
    }
    vec3 jamb = brick ? base * 0.9 : trim * 0.9;
    vec3 colour = fWindow(wall, jamb, frame, f, id, fw, lo, hi, brick ? 0.11 : 0.22, sash, fine, 0.10, lit);

    // The base course and the cornice.
    if (ground) {
      float plinth = fCover(f.y, 0.0, 0.085, fw.y);
      colour = mix(colour, base * 0.5 + vec3(0.025), plinth * trimVis);
    }
    float cornice = fCover(topDist, 0.0, brick ? 0.10 : 0.14, fw.y) * step(0.0, 8.0 - topDist);
    float eaves = fCover(topDist, brick ? 0.10 : 0.14, brick ? 0.16 : 0.22, fw.y) * step(0.0, 8.0 - topDist);
    colour *= 1.0 - 0.45 * eaves * trimVis;
    colour = mix(colour, trim, cornice * trimVis);

    return colour;
  }

  // Whether a point of a cell lies inside an arch (metres, so that the semicircle is round): a box from y0 to the springing line
  // and a semicircle over it up to y1, centred on cx and halfW wide each side, all in cell fractions. 1 inside, 0 outside, smooth at the edge.
  float fArch(vec2 f, vec2 fw, float cx, float halfW, float y0, float y1) {
    float r = halfW * BAY;
    vec2 p = vec2((f.x - cx) * BAY, f.y * STOREY);
    float yc = y1 * STOREY - r;
    float dBox = max(abs(p.x) - r, max(y0 * STOREY - p.y, p.y - yc));
    float dCirc = length(p - vec2(0.0, yc)) - r;
    float px = max(max(fw.x * BAY, fw.y * STOREY), 1e-4);
    return clamp(0.5 - min(dBox, dCirc) / px, 0.0, 1.0);
  }

  // The outside of a ballpark: brick with lighter pilasters at the bay lines, an arched steel gate in every third bay of the ground
  // storey (a recessed brick panel in the others), tall arched windows above, a dark base course, a band under the cornice.
  vec3 fBallpark(vec3 base, vec2 f, vec2 id, vec2 fw, float floors, float topDist, float seed, float fine) {
    bool ground = id.y < 0.5 && !fRaised;
    bool topRow = floors > 0.5 && id.y > floors - 1.5;
    vec3 wall = fCourses(base, f, id, fw, seed, true, ground);
    float trimVis = 1.0 - smoothstep(0.08, 0.22, max(fw.x, fw.y));
    float pilaster = clamp(fBand(f.x, -0.07, 0.07, fw.x), 0.0, 1.0) * trimVis;
    wall = mix(wall, base * 0.95 + vec3(0.02), pilaster * 0.55);
    vec3 trim = vec3(0.46, 0.44, 0.40);
    vec3 colour = wall;
    fGlass = 0.0;
    fEmit = vec3(0.0);
    bool gate = ground && mod(id.x + floor(seed * 3.0), 3.0) < 0.5;
    if (gate) {
      float hole = fArch(f, fw, 0.5, 0.31, 0.0, 0.82);
      float ring = fArch(f, fw, 0.5, 0.35, 0.0, 0.88) - hole;
      float bars = fBand(f.x * 6.0 + 0.5, -0.07, 0.07, fw.x * 6.0);
      vec3 steel = mix(vec3(0.012, 0.055, 0.035), vec3(0.035, 0.11, 0.07), clamp(bars, 0.0, 1.0) * trimVis);
      colour = mix(colour, trim, clamp(ring, 0.0, 1.0) * trimVis);
      colour = mix(colour, steel, hole);
    } else if (ground) {
      float panel = fBox(f, vec2(0.17, 0.12), vec2(0.83, 0.74), fw);
      colour *= 1.0 - 0.22 * panel * trimVis;
    } else {
      float hole = fArch(f, fw, 0.5, 0.19, 0.20, topRow ? 0.74 : 0.80);
      float ring = fArch(f, fw, 0.5, 0.225, 0.17, topRow ? 0.775 : 0.835) - hole;
      colour = mix(colour, trim, clamp(ring, 0.0, 1.0) * trimVis);
      vec3 pane = fInterior(id, 0.0);
      float mullion = fBox(f, vec2(0.485, 0.0), vec2(0.515, 1.0), fw) * fine;
      colour = mix(colour, mix(pane, vec3(0.05, 0.055, 0.05), mullion), hole);
      fGlass = hole * (1.0 - mullion);
      fEmit = fReflection(0.10, id, vec3(1.0)) * fGlass;
    }
    if (ground) {
      float plinth = fCover(f.y, 0.0, 0.10, fw.y);
      colour = mix(colour, base * 0.35 + vec3(0.012), plinth * trimVis);
    }
    if (topRow) {
      float band = fBox(f, vec2(-1.0, 0.85), vec2(2.0, 0.97), fw);
      colour = mix(colour, base * 0.45 + vec3(0.0, 0.012, 0.006), band * trimVis);
    }
    float cornice = fCover(topDist, 0.0, 0.10, fw.y) * step(0.0, 8.0 - topDist);
    float eaves = fCover(topDist, 0.10, 0.16, fw.y) * step(0.0, 8.0 - topDist);
    colour *= 1.0 - 0.45 * eaves * trimVis;
    return mix(colour, trim, cornice * trimVis);
  }

  // Concrete and stone towers: bands of glass between precast spandrels.
  vec3 fRibbon(vec3 base, vec2 f, vec2 id, vec2 fw, float floors, float topDist, float seed, float detail, float fine) {
    bool ground = id.y < 0.5 && !fRaised;
    vec3 wall = base;
    // Panel joints at the bay and the storey line.
    float joints = fBox(f, vec2(-1.0, 0.0), vec2(0.012, 1.0), fw) + fBox(f, vec2(0.0, -1.0), vec2(1.0, 0.012), fw) * 0.6;
    wall *= 1.0 - 0.22 * clamp(joints, 0.0, 1.0) * (1.0 - smoothstep(0.08, 0.22, max(fw.x, fw.y)));
    wall *= mix(1.0, mix(0.95, 1.05, fHash(id + seed)), fVary);
    vec2 lo = vec2(0.0, ground ? 0.04 : 0.27);
    vec2 hi = vec2(1.0, ground ? 0.94 : 0.77);
    vec3 frame = vec3(0.05, 0.055, 0.06);
    vec3 colour = fWindow(wall, base * 0.7, frame, f, id, fw, lo, hi, 0.12, 2.0, fine, 0.16, ground ? 1.0 : 0.0);
    float cap = fCover(topDist, 0.0, 0.12, fw.y) * step(0.0, 8.0 - topDist);
    colour = mix(colour, base * 0.6, cap);
    return colour;
  }

  // Glass towers: a curtain wall of mullions and tinted panes, spandrel at every floor slab.
  vec3 fCurtain(vec3 base, vec2 f, vec2 id, vec2 fw, float floors, float topDist, float seed, float detail, float fine) {
    bool ground = id.y < 0.5 && !fRaised;
    bool plant = floors > 9.5 && id.y > floors - 1.5;       // the top storey of a tall tower houses plant, behind louvres
    float column = floor(f.x * 2.0);                         // two panes to a bay
    vec2 pf = vec2(fract(f.x * 2.0), f.y);
    vec2 pw = vec2(fw.x * 2.0, fw.y);
    vec2 pid = vec2(id.x * 2.0 + column, id.y);
    float slab = ground ? 0.05 : 0.27;
    vec3 metal = seed > 0.5 ? vec3(0.38, 0.39, 0.41) : vec3(0.045, 0.05, 0.055);
    vec3 tint = mix(vec3(1.0), base / max(dot(base, vec3(0.3333)), 1e-3), 0.7);
    float spandrel = fBox(pf, vec2(-1.0, 0.0), vec2(2.0, slab - 0.012), pw);
    float mullion = 1.0 - fBox(pf, vec2(0.05, -1.0), vec2(0.95, 2.0), pw);
    float transom = fBox(pf, vec2(-1.0, slab - 0.012), vec2(2.0, slab + 0.012), pw);
    float frameShare = clamp(mullion + transom, 0.0, 1.0);
    // Whole floors are lit differently behind the glass, as blinds are drawn at different levels.
    float floorVary = mix(1.0, mix(0.88, 1.12, fHash(vec2(id.y, seed * 11.0 + 4.0))), fVary);
    vec3 pane = fInterior(pid, -0.5) * 0.5;
    float vision = (1.0 - spandrel) * (1.0 - frameShare);
    vec3 colour = mix(pane, base * 0.42, spandrel);
    colour = mix(colour, metal, frameShare);
    vec3 reflection = fReflection(0.24, pid, tint) * floorVary;
    vec3 emit = reflection * (vision + 0.35 * spandrel * (1.0 - frameShare));
    float glass = vision;
    if (plant) {
      float slats = fBand(f.y * 7.0, 0.0, 0.5, fw.y * 7.0);
      colour = mix(base * 0.5, base * 0.22, slats);
      emit = vec3(0.0);
      glass = 0.0;
    }
    fGlass = glass;
    fEmit = emit;
    return colour;
  }

  vec3 facadeColour(vec3 base) {
    vec2 raw = vUv;
    vec2 fw = fwidth(raw) + 1e-5;
    float floors = floor(raw.y / 64.0 + 0.004);   // the number of storeys in this wall; 0 if not known
    float v = raw.y - 64.0 * floors;              // storeys up the wall
    float wallSeed = floor((raw.x + 0.5) / 256.0);
    float special = floor(wallSeed / 8.0);        // 8 or more: a special style of wall (mesh.py FACADE_STYLE_STRIDE); 1 is a ballpark's outside
    wallSeed -= 8.0 * special;
    fRaised = wallSeed > 3.5;                     // seeds 4 to 7 mark a wall that starts above the ground (mesh.py FACADE_SEEDS)
    if (fRaised) wallSeed -= 4.0;

    // The view in the wall's own axes (T along the wall, up, N out of it), from the screen-space derivatives of position.
    vec3 N = normalize(cross(dFdx(vViewPosition), dFdy(vViewPosition)));
    vec3 up = normalize(mat3(viewMatrix) * vec3(0.0, 1.0, 0.0));
    vec3 T = normalize(cross(up, N));
    vec3 V = normalize(vViewPosition);
    fView = vec3(dot(V, T), dot(V, up), dot(V, N));
    fSky = dot(reflect(-V, N), up);
    fGlass = 0.0;
    fEmit = vec3(0.0);
    fVary = 0.0;

    // Near the ground the street shades the wall.
    float shade = (fRaised || floors < 0.5) ? 1.0 : mix(0.84, 1.0, smoothstep(0.0, 1.6, v));

    if (fw.x < 2e-5) return base * shade;   // a wall too short for a bay stays blank

    vec2 cell = vec2(raw.x, v);
    vec2 id = floor(cell);
    vec2 f = fract(cell);
    float detail = 1.0 - smoothstep(0.10, 0.40, max(fw.x, fw.y));
    float fine = 1.0 - smoothstep(0.03, 0.09, max(fw.x, fw.y));
    fVary = clamp((detail - 0.25) / 0.6, 0.0, 1.0);
    float topDist = floors > 0.5 ? floors - v : 99.0;
    // One number per wall, from its colour and its seed. The colour arrives interpolated, so it is rounded back to its
    // bytes first: the hash is chaotic, and a rounding error in a float would flip choices from pixel to pixel.
    float seed = fHash(floor(base.rg * 255.0 + 0.5) * 0.37 + wallSeed * 5.0);

    // What kind of wall is it? Reddish is brick; cool grey-blue and tall is glass; the rest is stone or concrete.
    float brick = smoothstep(1.4, 1.8, base.r / max(base.g, 0.002));
    float cool = smoothstep(1.03, 1.22, base.b / max(base.r, 0.002)) * (1.0 - brick);
    float glassWall = cool * smoothstep(6.5, 9.5, floors);

    // The average of each kind of wall, for when a cell is only a few pixels across: wall and window mixed by their share.
    vec3 tint = mix(vec3(1.0), base / max(dot(base, vec3(0.3333)), 1e-3), 0.7);
    float share = 0.21;
    vec3 farWall = brick > 0.5 ? mix(base * 0.84, fMortar(base), 0.22) : base * 0.98;   // brick and its mortar, in their shares
    float f0 = 0.10;
    int style = 0;
    if (glassWall > 0.5) {
      style = 2; share = 0.72; farWall = base * 0.42; f0 = 0.24;
    } else if (brick < 0.5 && floors > 7.5) {
      style = 1; share = 0.50; farWall = base; f0 = 0.16;
    }
    vec3 farColour = mix(farWall, vec3(0.03), share * 0.85);
    vec3 farEmit = fSkyColour(fSky) * mix(f0, 0.85, pow(1.0 - clamp(fView.z, 0.0, 1.0), 4.0)) * share * 0.85 * (style == 2 ? tint : vec3(1.0));

    vec3 nearColour = farColour;
    if (detail > 0.004) {
      if (special > 0.5) {
        nearColour = fBallpark(base, f, id, fw, floors, topDist, seed, fine);
      } else if (style == 2) {
        nearColour = fCurtain(base, f, id, fw, floors, topDist, seed, detail, fine);
      } else if (style == 1) {
        nearColour = fRibbon(base, f, id, fw, floors, topDist, seed, detail, fine);
      } else {
        nearColour = fPunched(base, f, id, fw, floors, v, topDist, seed, brick > 0.5, detail, fine);
      }
    } else {
      fGlass = 0.0;
      fEmit = vec3(0.0);
    }
    vec3 colour = mix(farColour, nearColour, detail);
    fEmit = mix(farEmit, fEmit, detail);
    fGlass = mix(share * 0.85, fGlass, detail);
    return colour * shade;
  }
`;

/** Draw windows, brick, trim and glass into the walls of a building, from the wall's UVs and colour. */
export function facade(material: MeshStandardMaterial): void {
  material.defines = { ...material.defines, USE_UV: '' };
  material.onBeforeCompile = (shader) => {
    shader.fragmentShader = shader.fragmentShader
      .replace('void main() {', `${FACADE_FUNCTIONS}\nvoid main() {`)
      .replace('#include <color_fragment>', '#include <color_fragment>\ndiffuseColor.rgb = facadeColour(diffuseColor.rgb);')
      .replace(
        '#include <roughnessmap_fragment>',
        '#include <roughnessmap_fragment>\nroughnessFactor = mix(roughnessFactor, 0.2, fGlass);',
      )
      .replace('#include <emissivemap_fragment>', '#include <emissivemap_fragment>\ntotalEmissiveRadiance += fEmit;');
  };
}
