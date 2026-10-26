"""Small desktop GUI (tkinter): pick an audio file, see the predicted emotion and probability bars."""
import threading
import tkinter as tk
from tkinter import filedialog, ttk
from pathlib import Path

from predict_emotion import MODELS, ROBUST, Predictor, record

COLORS = {"angry": "#d9534f", "disgust": "#7b8f2a", "fear": "#6f42c1", "happy": "#f0ad4e",
          "neutral": "#6c757d", "sad": "#0275d8", "surprise": "#e83e8c"}


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Speech emotion recognition (WavLM)")
        self.geometry("520x420")
        self.predictor = None
        self.status = tk.StringVar(value="Loading model...")
        bar = ttk.Frame(self); bar.pack(pady=10)
        ttk.Button(bar, text="Open audio file...", command=self.choose).pack(side="left", padx=5)
        ttk.Button(bar, text="Record 5 s (microphone)", command=self.record).pack(side="left", padx=5)
        self.result = tk.Label(self, text="", font=("Segoe UI", 22, "bold"))
        self.result.pack()
        tk.Label(self, textvariable=self.status, fg="gray").pack()
        self.canvas = tk.Canvas(self, width=480, height=260, highlightthickness=0)
        self.canvas.pack(pady=10)
        threading.Thread(target=self.load, daemon=True).start()

    def load(self) -> None:
        self.predictor = Predictor(heads_dir=ROBUST if ROBUST.exists() else MODELS)
        self.status.set("Model ready. Open a .wav/.flac/.mp3 file.")

    def choose(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.flac *.ogg *.mp3 *.m4a *.aiff")])
        if path and self.predictor:
            self.status.set("Analysing...")
            threading.Thread(target=self.run, args=(Path(path),), daemon=True).start()

    def record(self) -> None:
        if self.predictor:
            self.status.set("Recording 5 s ... speak now")
            threading.Thread(target=self.record_and_run, daemon=True).start()

    def record_and_run(self) -> None:
        try:
            path = record(5.0)
        except Exception as e:
            self.status.set(f"Recording failed: {e}"); return
        self.status.set("Analysing...")
        self.run(path)

    def run(self, path: Path) -> None:
        try:
            r = self.predictor.predict(path, 0.5)
        except Exception as e:
            self.status.set(f"Error: {e}"); return
        self.after(0, self.show, r, path)

    def show(self, r: dict, path: Path) -> None:
        text = f"{r['emotion'].upper()} {r['confidence']:.0%}" + ("  (uncertain)" if r["uncertain"] else "")
        self.result.config(text=text, fg=COLORS[r["emotion"]])
        self.status.set(f"{path.name} - {r['duration_s']} s")
        self.canvas.delete("all")
        for i, (k, v) in enumerate(r["probabilities"].items()):
            y = 8 + i * 35
            self.canvas.create_text(5, y + 12, anchor="w", text=k)
            self.canvas.create_rectangle(80, y, 80 + 380 * v, y + 24, fill=COLORS[k], outline="")
            self.canvas.create_text(84 + 380 * v, y + 12, anchor="w", text=f"{v:.0%}")


if __name__ == "__main__":
    App().mainloop()
