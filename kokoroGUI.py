import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import torch
from models import build_model
from kokoro import generate
import os
import sounddevice as sd
import gc  # For garbage collection
import threading  # For multithreading
import time  # For tracking playback time


class Kokoro:
    def __init__(self, model_path, device='cuda', voice_name='af', speed=1.2):
        self.device = device
        self.voice_name = voice_name
        self.speed = speed
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        self.model = build_model(model_path, self.device)
        self.voicepack = torch.load(f'voices/{self.voice_name}.pt', weights_only=True).to(self.device)

    def generate_audio(self, text):
        audio, out_ps = generate(self.model, text, self.voicepack, lang=self.voice_name[0], speed=self.speed, device=self.device)
        return audio

    def play_audio(self, audio):
        sd.play(audio, samplerate=24000)
        sd.wait()

    def generate_and_play_audio(self, text):
        audio = self.generate_audio(text)
        self.play_audio(audio)

    def clear_memory(self):
        """Clear memory by deleting model and voicepack, and calling garbage collection."""
        del self.model
        del self.voicepack
        if self.device == 'cuda':
            torch.cuda.empty_cache()  # Clear GPU memory
        gc.collect()  # Force garbage collection


class KokoroApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Kokoro Text-to-Speech")
        self.root.geometry("540x520")

        # Center the window and position it 30 pixels down from the top
        self.center_window(30)

        self.kokoro = None
        self.model_path = None  
        self.playback_thread = None
        self.stop_playback = False

        # Playback state variables
        self.playback_state = "stopped"     # Can be "playing", "paused", or "stopped"
        self.current_position = 0           # Current position in the audio data
        self.start_timestamp = 0            # Timestamp when playback starts
        self.accumulated_time = 0           # Total time played so far
        self.audio_data = None              # To store the generated audio data

        # Model Selection
        self.model_frame = ttk.LabelFrame(root, text="Model Selection")
        self.model_frame.pack(pady=10, padx=10, fill="x")

        self.model_label = ttk.Label(self.model_frame, text="No model selected")
        self.model_label.pack(side="left", padx=5)

        ttk.Button(self.model_frame, text="Browse", command=self.browse_model).pack(side="left", padx=5)
        ttk.Button(self.model_frame, text="Release", command=self.release_model).pack(side="left", padx=5)

        # Device Selection with callback
        self.device_var = tk.StringVar(value="cuda")
        self.device_var.trace_add("write", self.on_device_change)
        self.device_frame = ttk.LabelFrame(root, text="Device Selection")
        self.device_frame.pack(pady=10, padx=10, fill="x")

        ttk.Radiobutton(self.device_frame, text="CUDA", variable=self.device_var, value="cuda").pack(side="left", padx=5)
        ttk.Radiobutton(self.device_frame, text="CPU", variable=self.device_var, value="cpu").pack(side="left", padx=5)

        # Voice Selection with callback
        self.voice_var = tk.StringVar(value="af")
        self.voice_var.trace_add("write", self.on_voice_change)
        self.voice_frame = ttk.LabelFrame(root, text="Voice Selection")
        self.voice_frame.pack(pady=10, padx=10, fill="x")

        self.voice_options = [
            'af', 'af_bella', 'af_sarah', 'am_adam', 'am_michael',
            'bf_emma', 'bf_isabella', 'bm_george', 'bm_lewis',
            'af_nicole', 'af_sky'
        ]
        ttk.Combobox(self.voice_frame, textvariable=self.voice_var, values=self.voice_options, state="readonly").pack(padx=5, fill="x", expand=True)

        # Speed Control with callback
        self.speed_var = tk.DoubleVar(value=1.2)
        self.speed_var.trace_add("write", self.on_speed_change)
        self.speed_frame = ttk.LabelFrame(root, text="Speech Speed Control")
        self.speed_frame.pack(pady=10, padx=10, fill="x")

        ttk.Scale(self.speed_frame, from_=0.5, to=2.0, variable=self.speed_var, orient="horizontal").pack(padx=5, fill="x", expand=True)

        # Text Input
        self.text_frame = ttk.LabelFrame(root, text="Enter Text")
        self.text_frame.pack(pady=10, padx=10, fill="both", expand=True)

        self.text_input = tk.Text(self.text_frame, height=10)
        self.text_input.pack(padx=5, pady=5, fill="both", expand=True)

        # Buttons
        self.button_frame = ttk.Frame(root)
        self.button_frame.pack(pady=10, padx=10, fill="x")

        ttk.Button(self.button_frame, text="Speak", command=self.start_generate_and_play_thread).pack(side="left", padx=5)
        ttk.Button(self.button_frame, text="Stop", command=self.stop_playback_thread).pack(side="left", padx=5)
        ttk.Button(self.button_frame, text="Pause/Resume", command=self.toggle_playback).pack(side="left", padx=5)
        ttk.Button(self.button_frame, text="Exit", command=self.on_exit).pack(side="right", padx=5)

    def center_window(self, y_offset=0):
        """Center the window on the screen and position it y_offset pixels down from the top."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = y_offset
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def browse_model(self):
        """Open a file dialog to select a model file and load it immediately."""
        model_path = filedialog.askopenfilename(
            title="Select a Model File",
            filetypes=[("PyTorch Model Files", "*.pth"), ("All Files", "*.*")]
        )
        if model_path:
            try:
                self.model_path = model_path
                self.model_label.config(text=os.path.basename(model_path))
                # Initialize Kokoro with the selected model
                self.load_model_and_voice()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {e}")
                self.model_path = None
                self.model_label.config(text="No model selected")

    def load_model_and_voice(self):
        """Load model and voice pack with current settings."""
        if self.model_path:
            try:
                if self.kokoro is not None:
                    self.kokoro.clear_memory()
                
                device = self.device_var.get()
                voice_name = self.voice_var.get()
                speed = self.speed_var.get()
                
                self.kokoro = Kokoro(
                    self.model_path,
                    device=device,
                    voice_name=voice_name,
                    speed=speed
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model/voice: {e}")
                self.kokoro = None

    def release_model(self):
        """Release the currently selected model and clear memory."""
        if self.kokoro is not None:
            self.kokoro.clear_memory()
            self.kokoro = None
        self.model_path = None
        self.model_label.config(text="No model selected")

    def on_device_change(self, *args):
        """Handle device change."""
        if self.kokoro is not None:
            self.load_model_and_voice()

    def on_voice_change(self, *args):
        """Handle voice change."""
        if self.kokoro is not None:
            self.load_model_and_voice()

    def on_speed_change(self, *args):
        """Handle speed change."""
        if self.kokoro is not None:
            self.kokoro.speed = self.speed_var.get()

    def start_generate_and_play_thread(self):
        """Start the audio generation and playback in a separate thread."""
        if not self.model_path:
            messagebox.showwarning("Model Error", "Please select a model file.")
            self.browse_model()
            return

        text = self.text_input.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showwarning("Input Error", "Please enter some text.")
            return

        # Disable the button to prevent multiple clicks
        self.button_frame.winfo_children()[0].config(state="disabled")

        self.playback_thread = threading.Thread(target=self.generate_and_play, args=(text,), daemon=True)
        self.playback_thread.start()

    def generate_and_play(self, text):
        """Generate and play audio (to be run in a separate thread)."""
        try:
            if self.kokoro is None:
                self.load_model_and_voice()
            
            self.stop_playback = False
            self.audio_data = self.kokoro.generate_audio(text)

            # Start playback from the beginning
            self.current_position = 0
            self.accumulated_time = 0
            self.playback_state = "playing"
            self.start_timestamp = time.time()
            sd.play(self.audio_data[self.current_position:], samplerate=24000)
            sd.wait()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
        finally:
            self.root.after(0, lambda: self.button_frame.winfo_children()[0].config(state="normal"))

    def toggle_playback(self):
        """Toggles audio playback between playing and paused states."""
        if self.playback_state == "playing":
            sd.stop()
            end_timestamp = time.time()
            self.accumulated_time += end_timestamp - self.start_timestamp
            self.current_position = int(24000 * self.accumulated_time)
            self.playback_state = "paused"
            self.button_frame.winfo_children()[2].config(text="Resume")

        elif self.playback_state == "paused":
            sd.play(self.audio_data[self.current_position:], samplerate=24000)
            self.start_timestamp = time.time()
            self.playback_state = "playing"
            self.button_frame.winfo_children()[2].config(text="Pause")

    def stop_playback_thread(self):
        """Stop the audio playback."""
        self.stop_playback = True
        sd.stop()
        self.playback_state = "stopped"
        self.current_position = 0
        self.accumulated_time = 0
        self.button_frame.winfo_children()[2].config(text="Pause/Resume")

    def on_exit(self):
        """Clear memory before exiting the application."""
        if self.kokoro is not None:
            self.kokoro.clear_memory()
        self.root.quit()


if __name__ == '__main__':
    root = tk.Tk()
    app = KokoroApp(root)
    root.mainloop()
