import { CylinderGeometry, Group, Mesh, MeshBasicMaterial, type Scene, SphereGeometry, type Vector3 } from 'three';
import {
  MAX_NOTE_LENGTH,
  type Note,
  type NoteContext,
  describePlace,
  describeTarget,
  fetchTickets,
  issueUrl,
  loadNotes,
  newNoteId,
  saveNotes,
} from './notes';

/** What the notepad needs from the rest of the viewer. */
export interface NotepadHost {
  scene: Scene;
  /** Where the player stands and what is straight ahead, right now. */
  capture(): Omit<NoteContext, 'boston_note' | 'id'>;
  /** Move the player to a note's position and view. */
  go(context: NoteContext): void;
  notice(text: string): void;
  /** Free the mouse and drop held keys, because a form is about to take focus. */
  releaseInput(): void;
}

const MINE = new MeshBasicMaterial({ color: '#f2a93b' });
const TICKET = new MeshBasicMaterial({ color: '#4fb3ff' });
const POLE = new CylinderGeometry(0.04, 0.04, 2.6, 6).translate(0, 1.3, 0);
const HEAD = new SphereGeometry(0.22, 12, 8).translate(0, 2.75, 0);
const HINT_RANGE = 14; // metres within which a note's text is shown

function element<T extends HTMLElement>(id: string): T {
  return document.getElementById(id) as T;
}

function button(label: string, onClick: () => void): HTMLButtonElement {
  const node = document.createElement('button');
  node.type = 'button';
  node.textContent = label;
  node.addEventListener('click', onClick);
  return node;
}

/** Scene position of a note's marker: where the author stood, or what they pointed at from the air. */
function markerPosition(context: NoteContext): [number, number, number] {
  const spot = context.standing.flying && context.looking_at ? context.looking_at : context.standing;
  return [spot.x, Math.max(spot.z, -0.6), -spot.y];
}

/** Taking notes, listing them, and showing them as markers in the world. */
export class Notepad {
  private mine: Note[] = loadNotes();
  private tickets: Note[] = [];
  private pending: NoteContext | null = null;
  private readonly markers = new Group();
  private readonly form = element<HTMLFormElement>('note');
  private readonly text = element<HTMLTextAreaElement>('note-text');
  private readonly list = element('notes');

  constructor(private readonly host: NotepadHost) {
    host.scene.add(this.markers);
    this.text.maxLength = MAX_NOTE_LENGTH;

    element('note-open').addEventListener('click', () => this.open());
    element('notes-toggle').addEventListener('click', () => this.toggleList());
    element('note-save').addEventListener('click', () => this.finish(false));
    element('note-cancel').addEventListener('click', () => this.close());
    element('notes-download').addEventListener('click', () => this.download());
    this.form.addEventListener('submit', (event) => {
      event.preventDefault();
      this.finish(true);
    });
    this.form.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') this.close();
      else if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) this.finish(true);
    });

    this.refresh();
    void fetchTickets().then((tickets) => {
      this.tickets = tickets;
      // A note of mine that now exists as a ticket is marked as filed.
      for (const note of this.mine) {
        const filed = tickets.find((ticket) => ticket.context.id === note.context.id);
        if (filed) note.issue = filed.issue;
      }
      saveNotes(this.mine);
      this.refresh();
    });
  }

  get isOpen(): boolean {
    return !this.form.hidden;
  }

  /** Record the position and view at this instant, then ask what should be fixed. */
  open(): void {
    if (this.isOpen) return;
    this.pending = { boston_note: 1, id: newNoteId(), ...this.host.capture() };
    this.host.releaseInput();
    element('note-where').textContent =
      `${describePlace(this.pending)}. Looking at ${describeTarget(this.pending)}.`;
    this.list.hidden = true;
    this.form.hidden = false;
    this.text.value = '';
    this.text.focus();
  }

  /** Show the nearest note's text when the player walks up to its marker. */
  update(position: Vector3): void {
    let nearest: Note | undefined;
    let best = HINT_RANGE;
    for (const note of this.all()) {
      const [x, , z] = markerPosition(note.context);
      const distance = Math.hypot(x - position.x, z - position.z);
      if (distance < best) {
        best = distance;
        nearest = note;
      }
    }
    const hint = element('notehint');
    hint.textContent = nearest ? `${nearest.issue ? `Ticket #${nearest.issue.number}` : 'Your note'}: ${nearest.text}` : '';
  }

  /** My notes, then tickets from other people. */
  private all(): Note[] {
    const mine = new Set(this.mine.map((note) => note.context.id));
    return [...this.mine, ...this.tickets.filter((ticket) => !mine.has(ticket.context.id))];
  }

  private close(): void {
    this.form.hidden = true;
    this.pending = null;
    this.text.blur();
  }

  private finish(openTicket: boolean): void {
    const text = this.text.value.trim();
    if (!this.pending || !text) {
      this.text.focus();
      return;
    }
    const note: Note = { text, created: new Date().toISOString(), context: this.pending };
    this.mine.unshift(note);
    saveNotes(this.mine);
    this.close();
    this.refresh();
    if (openTicket) {
      window.open(issueUrl(note), '_blank', 'noopener');
      this.host.notice('Note saved. Finish the ticket in the GitHub tab that just opened.');
    } else {
      this.host.notice('Note saved on this device. Open Notes to turn it into a ticket.');
    }
  }

  private toggleList(): void {
    this.list.hidden = !this.list.hidden;
    if (!this.list.hidden) {
      this.close();
      this.host.releaseInput();
    }
  }

  private remove(note: Note): void {
    this.mine = this.mine.filter((other) => other !== note);
    saveNotes(this.mine);
    this.refresh();
  }

  private download(): void {
    const blob = new Blob([JSON.stringify(this.mine, null, 1)], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'boston-model-notes.json';
    link.click();
    URL.revokeObjectURL(link.href);
  }

  /** Redraw the list and the markers from the current notes. */
  private refresh(): void {
    const notes = this.all();
    element('notes-toggle').textContent = notes.length ? `Notes (${notes.length})` : 'Notes';
    element('notes-download').hidden = this.mine.length === 0;
    element('notes-empty').hidden = notes.length > 0;

    const rows = element('notes-list');
    rows.replaceChildren();
    for (const note of notes) {
      const mine = this.mine.includes(note);
      const row = document.createElement('li');
      const text = document.createElement('p');
      text.className = 'note-text';
      text.textContent = note.text;
      const meta = document.createElement('p');
      meta.className = 'note-meta';
      const status = note.issue ? `ticket #${note.issue.number}` : 'saved on this device only';
      meta.textContent = `${describePlace(note.context)} · ${status}`;
      const actions = document.createElement('div');
      actions.className = 'row';
      actions.append(
        button('Go there', () => {
          this.list.hidden = true;
          this.host.go(note.context);
        }),
      );
      if (note.issue) {
        const link = document.createElement('a');
        link.href = note.issue.url;
        link.target = '_blank';
        link.rel = 'noopener';
        link.textContent = 'Open ticket';
        actions.append(link);
      } else {
        actions.append(button('Create ticket', () => window.open(issueUrl(note), '_blank', 'noopener')));
      }
      if (mine) actions.append(button('Delete', () => this.remove(note)));
      row.append(text, meta, actions);
      rows.append(row);
    }

    this.markers.clear();
    for (const note of notes) {
      const material = note.issue ? TICKET : MINE;
      const pin = new Group();
      pin.add(new Mesh(POLE, material), new Mesh(HEAD, material));
      pin.position.set(...markerPosition(note.context));
      this.markers.add(pin);
    }
  }
}
