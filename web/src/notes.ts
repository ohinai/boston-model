import type { BuildingInfo } from './city';

/**
 * Notes: what someone walking the model wants fixed, with exactly where they
 * stood and what they were looking at. A note lives in the browser until it is
 * filed as a ticket (a GitHub issue), where people and tools pick it up.
 */

/** The GitHub repository tickets are filed in, as owner/name. */
export const REPO: string = import.meta.env.VITE_REPO || 'ohinai/boston-model';
export const MAX_NOTE_LENGTH = 1500;
const STORE_KEY = 'boston-model.notes.v1';

/** The position data stored with every note. Field names are a contract with AGENTS.md. */
export interface NoteContext {
  /** Format marker and version. Tools find a note's data block by this key. */
  boston_note: 1;
  /** Random id that ties a ticket back to the note saved in the author's browser. */
  id: string;
  /** Where the author stood: WGS84, world metres (x east, y north, z up), and view angles in degrees. */
  standing: {
    lat: number;
    lon: number;
    x: number;
    y: number;
    z: number;
    heading: number;
    pitch: number;
    flying: boolean;
  };
  /** The first thing straight ahead of the author, if anything was within range. */
  looking_at?: {
    lat: number;
    lon: number;
    x: number;
    y: number;
    z: number;
    distance: number;
    /** Tile layer hit: ground, green, water, roads, paths, rail, curbs, building_walls, building_roofs or trees. */
    layer: string;
    building?: Pick<BuildingInfo, 'id' | 'part_of' | 'name' | 'height' | 'height_source'>;
  };
  tile: [number, number];
  street?: string;
  near?: string;
  /** Opens the viewer at the same position and view. */
  link: string;
  /** Timestamp of the source data the author was looking at. */
  data?: string | null;
  /** Commit the viewer was built from, when known. */
  build?: string;
}

export interface Note {
  text: string;
  /** ISO timestamp. */
  created: string;
  context: NoteContext;
  /** Set once the note is known to exist as a ticket. */
  issue?: { number: number; url: string };
}

const LAYER_NAMES: Record<string, string> = {
  ground: 'paved ground',
  green: 'grass or planting',
  water: 'water',
  roads: 'a road',
  paths: 'a path',
  rail: 'rail track',
  curbs: 'a curb or bank',
  building_walls: 'a building wall',
  building_roofs: 'a roof',
  trees: 'a tree',
};

export function newNoteId(): string {
  return `n-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;
}

/** The address of the published viewer, used in ticket links even when running locally. */
export function siteUrl(): string {
  if (import.meta.env.VITE_SITE) return import.meta.env.VITE_SITE;
  const { hostname, origin, pathname } = window.location;
  if (hostname !== 'localhost' && hostname !== '127.0.0.1') return origin + pathname;
  const [owner, name] = REPO.split('/');
  return `https://${owner}.github.io/${name}/`;
}

/** "a building wall (Park Street Church), 23 m ahead" */
export function describeTarget(context: NoteContext): string {
  const target = context.looking_at;
  if (!target) return 'nothing in range';
  const what = LAYER_NAMES[target.layer] ?? target.layer;
  const name = target.building?.name ? ` (${target.building.name})` : '';
  return `${what}${name}, ${target.distance < 10 ? target.distance.toFixed(1) : target.distance.toFixed(0)} m ahead`;
}

/** "Lafayette Mall, near Brewer Fountain" */
export function describePlace(context: NoteContext): string {
  const parts = [context.street ?? 'unnamed ground'];
  if (context.near) parts.push(`near ${context.near}`);
  return parts.join(', ');
}

function issueTitle(note: Note): string {
  const first = note.text.trim().split('\n')[0];
  const title = first.length > 72 ? `${first.slice(0, 70).trimEnd()}…` : first;
  return note.context.street ? `${title} (${note.context.street})` : title;
}

/** The ticket text: the note, a line a person can read, and the data block tools read. */
export function issueBody(note: Note): string {
  const { context } = note;
  return [
    note.text.trim(),
    '',
    '---',
    `**Where:** ${describePlace(context)}. Looking at ${describeTarget(context)}.`,
    `**See it:** ${context.link}`,
    '',
    '<details>',
    '<summary>Position data (read by tools, please leave as it is)</summary>',
    '',
    '```json',
    JSON.stringify(context, null, 1),
    '```',
    '',
    '</details>',
  ].join('\n');
}

/** A link that opens GitHub's new-issue page with this note filled in. */
export function issueUrl(note: Note): string {
  const query = new URLSearchParams({ title: issueTitle(note), body: issueBody(note) });
  return `https://github.com/${REPO}/issues/new?${query}`;
}

function isContext(value: unknown): value is NoteContext {
  const context = value as NoteContext | null;
  return (
    typeof context === 'object' &&
    context !== null &&
    context.boston_note === 1 &&
    typeof context.id === 'string' &&
    Number.isFinite(context.standing?.x) &&
    Number.isFinite(context.standing?.y) &&
    Number.isFinite(context.standing?.z)
  );
}

/** Read a note back out of a ticket's text, or null if the ticket is not a note. */
export function parseIssueBody(body: string): { text: string; context: NoteContext } | null {
  for (const match of body.matchAll(/```json\s*([\s\S]*?)```/g)) {
    try {
      const context: unknown = JSON.parse(match[1]);
      if (isContext(context)) return { text: body.split(/\r?\n-{3,}\r?\n/)[0].trim(), context };
    } catch {
      // Not JSON, or not ours: keep looking.
    }
  }
  return null;
}

/** Open tickets that carry note data, so everyone sees what has already been reported. */
export async function fetchTickets(): Promise<Note[]> {
  try {
    // Filtering by position data rather than by label keeps this working before a ticket is labelled.
    const response = await fetch(`https://api.github.com/repos/${REPO}/issues?state=open&per_page=100`, {
      headers: { Accept: 'application/vnd.github+json' },
    });
    if (!response.ok) return [];
    const issues = (await response.json()) as {
      number: number;
      html_url: string;
      body: string | null;
      created_at: string;
      pull_request?: unknown;
    }[];
    const notes: Note[] = [];
    for (const issue of issues) {
      const parsed = issue.pull_request || !issue.body ? null : parseIssueBody(issue.body);
      if (parsed) notes.push({ ...parsed, created: issue.created_at, issue: { number: issue.number, url: issue.html_url } });
    }
    return notes;
  } catch {
    return []; // Offline, rate-limited or blocked: the viewer works without tickets.
  }
}

export function loadNotes(): Note[] {
  try {
    const stored: unknown = JSON.parse(window.localStorage.getItem(STORE_KEY) ?? '[]');
    return Array.isArray(stored) ? stored.filter((note: Note) => isContext(note?.context) && typeof note.text === 'string') : [];
  } catch {
    return [];
  }
}

export function saveNotes(notes: Note[]): void {
  try {
    window.localStorage.setItem(STORE_KEY, JSON.stringify(notes));
  } catch {
    // Storage can be full or disabled; the note still exists for this visit.
  }
}

export const buildCommit: string | undefined = import.meta.env.VITE_COMMIT || undefined;
