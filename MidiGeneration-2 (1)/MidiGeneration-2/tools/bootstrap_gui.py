import sys
import tkinter as tk
import io
from tkinter import filedialog
from mido import MidiFile
from tkinter import ttk
import random
import pretty_midi
from threading import Thread
import time
import pygame
import tempfile
from music21 import converter
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from music_transformer.generate import load_model, generate as transformer_generate
from music_transformer.vocabulary import start_token
from music_transformer.hparams import hparams, device
from music_transformer.generate import greedy_decode
from music_transformer.vocabulary import events_to_indices
from music_transformer.vocabulary import indices_to_events
from music_transformer.generate import audiate
from music_transformer.generate import list_parser
current_tempo = 120
model_path = "models/music_transformer.pt"
music_transformer = load_model(model_path)
# --- Colors ---
ORANGE = "#FFB366"
PINK = "#FF66B3"
YELLOW = "#EEFF33"
BLUE = "#6666FF"
CYAN = "#99FFFF"
BLACK = "#000000"
FONT_LABEL = ("Arial", 14, "bold")
FONT_BUTTON = ("Arial", 16, "bold")

entries = {}
generated_midi = None
visualizer_canvas = None
is_playing = False
playback_position = 0

# --- Root Setup ---
root = tk.Tk()
root.title("MIDI Generator")
root.attributes("-fullscreen", True)
root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))
root.geometry("600x500")
root.configure(bg="white")
user_settings = {"BPM": 120, "Key": "C major", "Bars": 4}
def draw_piano_roll(canvas, midi_data):
    global visualizer_notes
    canvas.delete("all")
    note_height = 10
    pixels_per_second = 100

    max_pitch = 0
    min_pitch = 127
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            max_pitch = max(max_pitch, note.pitch)
            min_pitch = min(min_pitch, note.pitch)

    height = (max_pitch - min_pitch + 1) * note_height
    canvas.config(scrollregion=(0, 0, canvas_width, height))

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            x0 = note.start * pixels_per_second
            x1 = note.end * pixels_per_second
            y0 = (max_pitch - note.pitch) * note_height
            y1 = y0 + note_height
            canvas.create_rectangle(x0, y0, x1, y1, fill="green", outline="black")

def generate_melody(bpm, key_str, num_notes=32):
    scale_notes = {
        "major": [0, 2, 4, 5, 7, 9, 11],
        "minor": [0, 2, 3, 5, 7, 8, 10]
    }

    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    parts = key_str.split()
    root_note = note_names.index(parts[0]) if len(parts) == 2 else 0
    mode = parts[1].lower() if len(parts) == 2 else "major"
    scale = [(root_note + interval) % 12 for interval in scale_notes.get(mode, scale_notes["major"])]

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    instrument = pretty_midi.Instrument(program=0)

    start_time = 0.0
    duration = 0.5
    for _ in range(num_notes):
        pitch_class = random.choice(scale)
        octave = random.choice([4, 5])
        pitch = pitch_class + 12 * octave
        note = pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=start_time + duration)
        instrument.notes.append(note)
        start_time += duration

    pm.instruments.append(instrument)
    return pm

def main_ui():
    global visualizer_canvas
    for widget in root.winfo_children():
        widget.destroy()

    root.configure(bg="white")

    top_frame = tk.Frame(root, height=250)
    top_frame.pack(fill="x")

    left_panel = tk.Frame(top_frame, bg=ORANGE, width=400, height=250)
    left_panel.pack(side="left", fill="both", expand=True)

    canvas_frame = tk.Frame(left_panel)
    canvas_frame.pack(fill="both", expand=True)

    x_scrollbar = tk.Scrollbar(canvas_frame, orient="horizontal")
    x_scrollbar.pack(side="bottom", fill="x")

    y_scrollbar = tk.Scrollbar(canvas_frame, orient="vertical")
    y_scrollbar.pack(side="right", fill="y")

    visualizer_canvas = tk.Canvas(
        canvas_frame, bg="black", highlightthickness=0,
        xscrollcommand=x_scrollbar.set,
        yscrollcommand=y_scrollbar.set
    )
    visualizer_canvas.pack(side="left", fill="both", expand=True)

    x_scrollbar.config(command=visualizer_canvas.xview)
    y_scrollbar.config(command=visualizer_canvas.yview)

    visualizer_canvas.create_text(200, 100, text="Load or generate MIDI to visualize", fill="white", font=("Arial", 14))

    right_panel = tk.Frame(top_frame, bg=PINK, width=200)
    right_panel.pack(side="right", fill="y", padx=10)

    form_labels = ["BPM", "Key", "Bars"]
    entries.clear()
    for label in form_labels:
        lbl = tk.Label(right_panel, text=label, font=FONT_LABEL, bg=PINK, fg=BLACK)
        lbl.pack(anchor="w", padx=10, pady=(5, 0))
        if label == "Key":
            key_options = [f"{note} {mode}" for note in ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] for mode in ['major', 'minor']]
            ent = ttk.Combobox(right_panel, values=key_options, font=FONT_LABEL, width=14)
            ent.set("C major")
        else:
            ent = tk.Entry(right_panel, bg=CYAN, font=FONT_LABEL, width=16, justify="left")
        ent.pack(padx=10, pady=(0, 5))
        entries[label] = ent

    def generate():
        global generated_midi
        bpm = user_settings["BPM"]
        key = user_settings["Key"]
        bars = user_settings["Bars"]

        tokens_per_bar = 64  # Estimate or fine-tune this value
        beats_per_bar = 4  # Assuming 4/4
        max_beats = bars * beats_per_bar

        print(f"Generating {bars} bars at {bpm} BPM in {key}...")

        token_ids = greedy_decode(
        model=music_transformer,
        inp=indices_to_events([start_token]),
        mode="categorical",
        temperature=1.0,
        k=5
        )          

        print(f"Generated {len(token_ids)} tokens.")
        print("First 20 tokens:", token_ids[:20])
        print("Last 20 tokens:", token_ids[-20:])
        try:
            print("Saving midi file at generated.mid...")
            audiate(token_ids, save_path="generated.mid", verbose=True)
            print("Done")
            generated_midi = pretty_midi.PrettyMIDI("generated.mid")
        except Exception as e:
            print("Failed to generate valid MIDI:", e)
            return


        if visualizer_canvas:
            draw_piano_roll(visualizer_canvas, generated_midi)

        play_midi_file(generated_midi)
        play_visualizer(visualizer_canvas, generated_midi, generated_midi.get_end_time())
        print("New transformer-based melody generated!")

    def save():
        if not generated_midi:
            print("No MIDI to save.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".mid", filetypes=[("MIDI files", "*.mid *.midi")])
        if file_path:
            generated_midi.write(file_path)
            print(f"MIDI saved to {file_path}")

    def load_midi():
        file_path = filedialog.askopenfilename(filetypes=[("MIDI files", "*.mid *.midi")])
        if file_path:
            print(f"Loaded MIDI file: {file_path}")
            midi = pretty_midi.PrettyMIDI(file_path)
            draw_piano_roll(visualizer_canvas, midi)

            tempo_changes, tempi = midi.get_tempo_changes()
            bpm = int(tempi[0]) if tempi.any() else "Unknown"

            try:
                score = converter.parse(file_path)
                key = score.analyze("key").name
            except:
                key = "Unknown"

            entries["BPM"].delete(0, tk.END)
            entries["BPM"].insert(0, str(bpm))
            entries["Key"].delete(0, tk.END)
            entries["Key"].insert(0, str(key))

            global current_tempo
            tempo_changes, tempi = midi.get_tempo_changes()
            current_tempo = int(tempi[0]) if len(tempi) else 120

    def settings():
        try:
            user_settings["BPM"] = int(entries["BPM"].get())
            user_settings["Key"] = entries["Key"].get()
            user_settings["Bars"] = int(entries["Bars"].get())
            print("Settings saved:", user_settings)
        except ValueError:
            print("Invalid input in settings.")

    def about():
        show_about()

    button_commands = {
        "Generate": generate,
        "Save": save,
        "Settings": settings,
        "About": about
    }

    for text in button_commands:
        btn = tk.Button(right_panel, text=text, bg="white", fg="black", font=FONT_BUTTON,
                        bd=2, relief="raised", activebackground="#DDDDDD", command=button_commands[text])
        btn.pack(pady=8, padx=10, fill="x")

    middle_frame = tk.Frame(root, bg=YELLOW, height=40)
    middle_frame.pack(fill="x")

    tk.Button(middle_frame, text="Load MIDI", bg=YELLOW, fg="black", font=("Arial", 12), bd=0, command=load_midi).pack(side="left", padx=20, pady=5)
    tk.Button(middle_frame, text="Play", bg=YELLOW, fg="black", font=("Arial", 12), bd=0,
          command=lambda: (
              play_midi_file(generated_midi),
              play_visualizer(visualizer_canvas, generated_midi, generated_midi.get_end_time())
          ) if generated_midi else None).pack(side="left", padx=20, pady=5)
    tk.Button(middle_frame, text="Stop", bg=YELLOW, fg="black", font=("Arial", 12), bd=0, command=stop_visualizer).pack(side="left", padx=20, pady=5)

    bottom_frame = tk.Frame(root, bg=BLUE)
    bottom_frame.pack(fill="both", expand=True)

    console = tk.Text(bottom_frame, bg="black", fg="lime", insertbackground="white", font=("Courier", 12), wrap="word", state="normal")
    console.pack(fill="both", expand=True, padx=10, pady=10)

    sys.stdout = TextRedirector(console)
    sys.stderr = TextRedirector(console)
    print("MIDI Generator Loaded!")

visualizer_notes = []
canvas_width = 2000

def play_midi_file(pm):
    global is_playing
    is_playing = True

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as temp_file:
        pm.write(temp_file.name)
        temp_path = temp_file.name

    pygame.mixer.init()
    pygame.mixer.music.load(temp_path)
    pygame.mixer.music.play(loops=-1)

    def check_end():
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        os.unlink(temp_path)

    Thread(target=check_end, daemon=True).start()

def play_visualizer(canvas, midi_data, duration):
    global is_playing, playback_position
    note_height = 10
    pixels_per_second = 100
    #seconds_per_beat = 60 / tempo
    fps = 30

    is_playing = True
    playback_position = 0

    playhead = canvas.create_line(0, -1000, 0, 2000, fill="red", width=2)

    def update():
        global playback_position
        start_time = time.time()
        while is_playing:
            elapsed = time.time() - start_time
            if elapsed > duration:
                start_time = time.time()
            x = int(elapsed * pixels_per_second)
            canvas.coords(playhead, x, 0, x, canvas.winfo_height())
            canvas.xview_moveto(x / canvas_width)
            time.sleep(1 / fps)

    Thread(target=update, daemon=True).start()

def stop_visualizer():
    global is_playing
    is_playing = False
    if pygame.mixer.get_init():
        pygame.mixer.music.stop()

def show_about():
    for widget in root.winfo_children():
        widget.destroy()
    about_frame = tk.Frame(root, bg=PINK)
    about_frame.pack(fill="both", expand=True)
    tk.Button(about_frame, text="Back", font=FONT_BUTTON, bg=PINK, fg=YELLOW, bd=0,
              activebackground=PINK, command=main_ui).place(x=10, y=10)
    tk.Label(about_frame, text="About", font=FONT_BUTTON, bg=PINK, fg=YELLOW).place(relx=0.5, y=10, anchor="n")

def show_settings():
    for widget in root.winfo_children():
        widget.destroy()
    settings_frame = tk.Frame(root, bg=PINK)
    settings_frame.pack(fill="both", expand=True)
    tk.Button(settings_frame, text="Back", font=FONT_BUTTON, bg=PINK, fg=YELLOW, bd=0,
              activebackground=PINK, command=main_ui).place(x=10, y=10)
    tk.Label(settings_frame, text="Settings", font=FONT_BUTTON, bg=PINK, fg=YELLOW).place(relx=0.5, y=10, anchor="n")

class TextRedirector(io.TextIOBase):
    def __init__(self, widget):
        self.widget = widget
    def write(self, s):
        self.widget.insert("end", s)
        self.widget.see("end")
    def flush(self):
        pass

main_ui()
root.mainloop()