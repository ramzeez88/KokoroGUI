# kokoroGUI.py

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from kokoro import KPipeline
import soundfile as sf
import sounddevice as sd
import threading
import time
import gc
import torch


class KokoroGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Kokoro Text-to-Speech")
        self.root.geometry("540x520")
        self.center_window(30)

        # Device selection (with callback)
        self.device_var = tk.StringVar(value='cpu')
        self.device_var.trace_add("write", self.on_device_change)
        self.device_frame = ttk.LabelFrame(self.root, text="Device Selection")
        self.device_frame.pack(fill="x", padx=10, pady=5)
        ttk.Radiobutton(self.device_frame, text="CPU", variable=self.device_var, value='cpu').pack(side="left", padx=5, pady=5)
        ttk.Radiobutton(self.device_frame, text="CUDA", variable=self.device_var, value='cuda').pack(side="left", padx=5, pady=5)

        # Voice selection
        self.voice_var = tk.StringVar(value='af_heart')
        self.voice_frame = ttk.LabelFrame(self.root, text="Voice Selection")
        self.voice_frame.pack(fill="x", padx=10, pady=5)
        self.voice_dropdown = ttk.Combobox(self.voice_frame, textvariable=self.voice_var, state="readonly")
        self.voice_dropdown['values'] = ('af_heart', 'af_bella', 'af_jessica', 'af_sarah', 'am_adam', 'am_michael',
                                          'bf_emma', 'bf_isabella', 'bm_george', 'bm_lewis', 'af_nicole', 'af_sky')
        self.voice_dropdown.pack(fill="x", padx=5, pady=5)

        # Speed selection
        self.speed_var = tk.DoubleVar(value=1.0)
        self.speed_frame = ttk.LabelFrame(self.root, text="Speed Selection")
        self.speed_frame.pack(fill="x", padx=10, pady=5)
        self.speed_scale = ttk.Scale(self.speed_frame, from_=0.5, to=2.0, variable=self.speed_var, orient="horizontal")
        self.speed_scale.pack(fill="x", padx=5, pady=5)

        # Text input
        self.text_frame = ttk.LabelFrame(self.root, text="Text Input")
        self.text_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.text_input = tk.Text(self.text_frame, height=10)
        self.text_input.pack(fill="both", expand=True, padx=5, pady=5)

        # Buttons
        self.button_frame = ttk.Frame(self.root)
        self.button_frame.pack(fill="x", padx=10, pady=5)
        self.play_button = ttk.Button(self.button_frame, text="Play", command=self.play_audio)
        self.play_button.pack(side="left", padx=5, pady=5)
        self.pause_button = ttk.Button(self.button_frame, text="Pause/Resume", command=self.pause_resume_audio)
        self.pause_button.pack(side="left", padx=5, pady=5)
        self.stop_button = ttk.Button(self.button_frame, text="Stop", command=self.stop_audio)
        self.stop_button.pack(side="left", padx=5, pady=5)
        self.exit_button = ttk.Button(self.button_frame, text="Exit", command=self.on_exit)
        self.exit_button.pack(side="right", padx=5, pady=5)

        # Audio control variables
        self.audio_thread = None
        self.is_playing = False
        self.is_paused = False
        self.stop_playback = False
        self.audio_data = None
        self.current_position = 0
        self.start_timestamp = 0
        self.accumulated_time = 0
        self.stream = None
        self.pipeline = None

      
        self.initialize_pipeline()

    def center_window(self, y_offset=0):
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = y_offset
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def initialize_pipeline(self):
        """Initializes or re-initializes the KPipeline."""
        self.clear_pipeline() 
        try:
            self.pipeline = KPipeline(lang_code='a', device=self.device_var.get())
        except RuntimeError as e:
            if "No CUDA GPUs are available" in str(e):
                messagebox.showerror("CUDA Error", "No CUDA GPUs are available.  Switching to CPU.")
                self.device_var.set('cpu')  # Switch to CPU
                self.pipeline = KPipeline(lang_code='a', device='cpu')
            else:
                messagebox.showerror("Initialization Error", str(e))
        except Exception as e:
            messagebox.showerror("Initialization Error", str(e))

    def on_device_change(self, *args):
        """Handles device changes immediately."""
        self.initialize_pipeline()


    def play_audio(self):
        if self.is_playing:
            messagebox.showwarning("Warning", "Audio is already playing.")
            return

        text = self.text_input.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter some text.")
            return

        self.play_button.config(state="disabled")
        self.is_playing = True
        self.is_paused = False
        self.stop_playback = False

        device = self.device_var.get()
        voice = self.voice_var.get()
        speed = self.speed_var.get()

        self.audio_thread = threading.Thread(target=self.generate_and_play_audio, args=(text, device, voice, speed))
        self.audio_thread.start()

    def generate_and_play_audio(self, text, device, voice, speed):
       
        generator = self.pipeline(text, voice=voice, speed=speed, split_pattern=r'\n+')

        self.audio_data = []
        try:
            for i, (gs, ps, audio) in enumerate(generator):
                if self.stop_playback:
                    break
                self.audio_data.append(audio)

            if not self.stop_playback:
                self.audio_data = [item for sublist in self.audio_data for item in sublist]
                self.audio_data = [float(x) for x in self.audio_data]

                self.current_position = 0
                self.accumulated_time = 0
                self.start_timestamp = time.time()

                self.stream = sd.OutputStream(
                    samplerate=24000,
                    channels=1,
                    callback=self.audio_callback,
                    blocksize=1024
                )
                self.stream.start()
                while self.current_position < len(self.audio_data) and not self.stop_playback:
                    sd.sleep(10)
                if self.stream:
                    self.stream.stop()
                    self.stream.close()
                    self.stream = None

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
        finally:
            self.is_playing = False
            self.root.after(0, lambda: self.play_button.config(state="normal"))

    def audio_callback(self, outdata, frames, time_info, status):
        if status:
            print(status)
        if self.is_paused or self.stop_playback:
            outdata[:] = 0
            return

        chunksize = min(len(self.audio_data) - self.current_position, frames)
        outdata[:chunksize, 0] = self.audio_data[self.current_position:self.current_position + chunksize]
        if chunksize < frames:
            outdata[chunksize:, 0] = 0
        self.current_position += chunksize

    def pause_resume_audio(self):
        if not self.is_playing:
            messagebox.showwarning("Warning", "No audio is currently playing.")
            return

        if self.is_paused:
            self.is_paused = False
            self.start_timestamp = time.time() - self.accumulated_time
            self.pause_button.config(text="Pause")
        else:
            self.is_paused = True
            self.accumulated_time = time.time() - self.start_timestamp
            self.pause_button.config(text="Resume")

    def stop_audio(self):
        if self.is_playing:
            self.stop_playback = True
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            self.is_playing = False
            self.is_paused = False
            self.current_position = 0
            self.accumulated_time = 0
            self.pause_button.config(text="Pause/Resume")
         

    def clear_pipeline(self):
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
            if self.device_var.get() == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

    def on_exit(self):
        self.stop_audio()
        self.clear_pipeline()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = KokoroGUI(root)
    root.mainloop()
