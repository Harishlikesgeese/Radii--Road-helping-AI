# GUI.py
"""
Road Hazard Dashboard (fixed + optimized)
- Optimized CameraWorker.run() to reduce lag (lower resolution, YOLO every N frames)
- Beep alert when a NEW critical hazard appears (Windows winsound; safe no-op on other OS)
- Critical cameras auto-clear after a cooldown (30s by default)
- Critical list popup with Refresh button
- Robust imports for analyzer and camconnect modules
- Comments included to explain key changes
- FIXED: Green bounding boxes now persist across frames (no more disappearing quickly)
"""

import os
import threading
import time
import math
import queue
import cv2
import customtkinter as ctk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk

# Try to import winsound for beep on Windows; otherwise provide a no-op fallback
try:
    import winsound
    def _beep():
        try:
            winsound.Beep(1000, 400)  # freq=1000Hz, duration=400ms
        except Exception:
            pass
except Exception:
    # Non-Windows fallback: try cross-platform beep via stdout bell (may be muted)
    def _beep():
        try:
            print("\a", end="", flush=True)
        except Exception:
            pass

# -----------------------
# Analyzer import handling
# -----------------------
analyzer = None
analyzer_module_names = [
    "Video_and_Image_Anylyzer",
    "Video and Image Anylyzer",
    "Video and Image Anylyzer.py",
    "Video_and_Image_Analyzer"
]
for name in analyzer_module_names:
    try:
        analyzer = __import__(name)
        break
    except Exception:
        pass

# If direct import failed, attempt to load from file path
if analyzer is None:
    try:
        import importlib.util
        filepath = os.path.join(os.getcwd(), "Video and Image Anylyzer.py")
        if os.path.exists(filepath):
            spec = importlib.util.spec_from_file_location("analyzer_custom", filepath)
            analyzer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(analyzer)
    except Exception:
        analyzer = None

# Last fallback: alternate filename
if analyzer is None and os.path.exists(os.path.join(os.getcwd(), "Video_and_Image_Anylyzer.py")):
    import importlib.util
    spec = importlib.util.spec_from_file_location("analyzer_custom2", os.path.join(os.getcwd(), "Video_and_Image_Anylyzer.py"))
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)

# -----------------------
# Camconnect import (optional)
# -----------------------
camconnect = None
try:
    import Camconnect as camconnect
except Exception:
    cam_path = os.path.join(os.getcwd(), "Camconnect.py")
    if os.path.exists(cam_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("camconnect_custom", cam_path)
        camconnect = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(camconnect)

# -----------------------
# Model access & hazards
# -----------------------
# If analyzer exists, try to access model and hazard maps
model = None
ROAD_HAZARDS = {}
FIX_MAP = {}
if analyzer is not None:
    model = getattr(analyzer, "model", None)
    ROAD_HAZARDS = getattr(analyzer, "ROAD_HAZARDS", {})
    FIX_MAP = getattr(analyzer, "FIX_MAP", {})

MODEL_AVAILABLE = model is not None

# -----------------------
# Critical hazard helper
# -----------------------
CRITICAL_KEYWORDS = ["crash", "accident", "🚨", "fire", "🔥", "smoke", "⚠️"]
def is_critical(hazard_text: str):
    """
    Decide whether a hazard text should be considered 'critical'.
    We look for keywords (case-insensitive) and emoji markers.
    """
    if not hazard_text:
        return False
    low = hazard_text.lower()
    for kw in CRITICAL_KEYWORDS:
        # check both lowercase substring and original (to catch emoji)
        if kw in low or kw in hazard_text:
            return True
    return False

# -----------------------
# CameraWorker: handles per-camera capture & inference
# -----------------------
class CameraWorker(threading.Thread):
    """
    Worker thread to capture frames from a video source and run detection.
    Optimizations included:
      - Lower camera resolution to 640x360 to reduce CPU load.
      - Run YOLO only every N frames (configurable below) to reduce inference frequency.
      - Keep queue size small and drop oldest frames when full to avoid GUI lag.
    FIXED: Persist last detected boxes/labels and draw them on EVERY frame to prevent quick disappearance.
    """
    def __init__(self, source, frame_queue, camera_id_label):
        super().__init__(daemon=True)
        self.source = source
        self.frame_queue = frame_queue  # queue.Queue() instance provided by GUI
        self.camera_id_label = camera_id_label
        self._stop = threading.Event()
        self.cap = None
        self.last_boxes = []     # List of (x1, y1, x2, y2) tuples from last detection
        self.last_labels = []    # List of labels from last detection

        # Tunable settings (small numbers for good responsiveness)
        self.target_width = 640
        self.target_height = 360
        self.target_fps = 30
        self.yolo_every_n_frames = 12  # run YOLO once every 12 frames

    def stop(self):
        self._stop.set()
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass

    def run(self):
        """
        Optimized capture loop:
          - Open capture (int index or URL)
          - Set reduced resolution
          - Increment frame counter and only call model.predict on every N-th frame
          - Maintain the frame_queue by dropping oldest frame if full, keeping only the latest
          - FIXED: Collect boxes/labels during YOLO runs, then draw LAST boxes on EVERY frame
        """
        try:
            try:
                src = int(self.source)
            except Exception:
                src = self.source

            self.cap = cv2.VideoCapture(src)

            # If cannot open, try briefly then report error
            if not self.cap.isOpened():
                time.sleep(0.2)
                if not self.cap.isOpened():
                    self.frame_queue.put((self.camera_id_label, None, f"[Error] Cannot open {self.source}"))
                    return

            # Lower camera resolution and fps to reduce processing cost
            try:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            except Exception:
                pass  # some capture sources may ignore these settings

            frame_counter = 0

            while not self._stop.is_set():
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    # quick sleep to avoid busy loop when frames are not available
                    time.sleep(0.01)
                    continue

                frame_counter += 1
                run_yolo = (frame_counter % self.yolo_every_n_frames == 0)

                overlay_frame = frame.copy()
                hazards_found = []
                current_boxes = []      # Temp lists for this YOLO run
                current_labels = []

                if MODEL_AVAILABLE and run_yolo:
                    try:
                        # Use analyzer/model to predict. Pass conf threshold if available in analyzer.
                        conf = getattr(analyzer, "SCORE_THRESH", 0.35)
                        results = model.predict(source=frame, conf=conf, verbose=False)

                        # Process boxes (if any)
                        for r in results:
                            if hasattr(r, "boxes") and r.boxes is not None:
                                for box_obj in r.boxes:
                                    # Safely get class id and label
                                    try:
                                        cls_id = int(box_obj.cls)
                                        label = r.names[cls_id]
                                    except Exception:
                                        # fallback for different object types
                                        try:
                                            cls_id = int(box_obj.cls.cpu().numpy())
                                            label = r.names[cls_id]
                                        except Exception:
                                            continue

                                    # bounding box coords
                                    try:
                                        xy = box_obj.xyxy[0]
                                        x1, y1, x2, y2 = map(int, xy)
                                    except Exception:
                                        continue

                                    # Collect for persistence (don't draw yet)
                                    current_boxes.append((x1, y1, x2, y2))
                                    current_labels.append(label)

                                    # If label corresponds to a hazard name, add text
                                    if label in ROAD_HAZARDS:
                                        hazards_found.append(ROAD_HAZARDS[label])

                        # Update last known boxes/labels after processing
                        self.last_boxes = current_boxes
                        self.last_labels = current_labels

                    except Exception as e:
                        # If model prediction fails, put an error overlay (non-fatal)
                        try:
                            cv2.putText(overlay_frame, f"Model error: {e}", (10, 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                        except Exception:
                            pass

                # FIXED: Always draw the LAST boxes/labels on overlay_frame (persists across frames)
                for i, (x1, y1, x2, y2) in enumerate(self.last_boxes):
                    label = self.last_labels[i] if i < len(self.last_labels) else "Unknown"
                    try:
                        cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(overlay_frame, f"{label}", (x1, y1 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    except Exception:
                        pass

                # Manage queue: if full, pop oldest to make room (keeps GUI lag low)
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except Exception:
                        pass

                # Put the newest frame into queue (overlay_frame in BGR)
                try:
                    self.frame_queue.put((self.camera_id_label, overlay_frame, hazards_found))
                except Exception:
                    pass

                # Small sleep to avoid burning CPU; inference cost is the dominating factor
                time.sleep(0.005)

        finally:
            try:
                if self.cap:
                    self.cap.release()
            except Exception:
                pass

# -----------------------
# Main GUI Application
# -----------------------
class RoadHazardApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Road Hazard Dashboard")
        self.geometry("1200x700")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")

        # ---------- State ----------
        self.camera_sources = []     # list of camera source strings/int
        self.camera_workers = {}     # label -> CameraWorker
        self.frame_queues = {}       # label -> queue.Queue
        self.current_page = 0
        self.per_page = 2

        # Critical camera tracking
        self.critical_cameras = set()     # set of camera numbers (1-based) currently considered critical
        self.critical_timers = {}         # camera_number -> last_time_seen (epoch seconds)
        self.critical_timeout = 30        # seconds to keep camera flagged after last critical detection

        # Build UI and start discovery
        self._build_layout()
        threading.Thread(target=self._initial_discover, daemon=True).start()
        # Start periodic UI update loop
        self.after(50, self._update_frames_periodic)

    # -----------------------
    # UI layout
    # -----------------------
    def _build_layout(self):
        top_frame = ctk.CTkFrame(self, height=60)
        top_frame.pack(side="top", fill="x", padx=6, pady=6)

        self.jump_btn = ctk.CTkButton(top_frame, text="Jump to Camera", command=self._jump_popup)
        self.jump_btn.pack(side="left", padx=(10,6))

        self.crit_btn = ctk.CTkButton(top_frame, text="Critical Cameras", command=self._show_critical_list)
        self.crit_btn.pack(side="left")

        main_frame = ctk.CTkFrame(self)
        main_frame.pack(side="top", fill="both", expand=True, padx=6, pady=(0,6))

        panels_frame = ctk.CTkFrame(main_frame)
        panels_frame.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        # Two camera panels per page
        self.cam_frames = []
        self.image_labels = []
        self.cam_name_labels = []

        for i in range(self.per_page):
            f = ctk.CTkFrame(panels_frame, width=400, height=360, corner_radius=6)
            f.pack(side="left", expand=True, fill="both", padx=(0 if i == 0 else 8, 0))
            f.pack_propagate(False)
            self.cam_frames.append(f)

            img_label = ctk.CTkLabel(f, text="", anchor="center")
            img_label.pack(expand=True, fill="both", padx=6, pady=6)
            self.image_labels.append(img_label)

            name_label = ctk.CTkLabel(f, text=f"Camera #{i+1}", anchor="w")
            name_label.pack(side="bottom", fill="x", padx=8, pady=6)
            self.cam_name_labels.append(name_label)

        # Page arrows
        arrow_frame = ctk.CTkFrame(self, width=60)
        arrow_frame.place(relx=0.98, rely=0.5, anchor="e")
        self.next_btn = ctk.CTkButton(arrow_frame, text="→", width=40, command=self._next_page)
        self.next_btn.pack(padx=4, pady=4)
        self.prev_btn = ctk.CTkButton(arrow_frame, text="←", width=40, command=self._prev_page)
        self.prev_btn.pack(padx=4, pady=4)

        # Bottom log
        self.log_box = ctk.CTkTextbox(self, height=110, corner_radius=6)
        self.log_box.pack(side="bottom", fill="x", padx=10, pady=8)
        self._log("App started. Model available: " + str(MODEL_AVAILABLE))

    # -----------------------
    # Logging helper
    # -----------------------
    def _log(self, text):
        ts = time.strftime("%H:%M:%S")
        try:
            self.log_box.insert("end", f"[{ts}] {text}\n")
            self.log_box.see("end")
        except Exception:
            print(f"[{ts}] {text}")

    # -----------------------
    # Camera discovery and adding
    # -----------------------
    def _initial_discover(self):
        self._log("Scanning local cameras...")
        cam_list = []
        if camconnect:
            try:
                cam_list = camconnect.check_local_cameras()
            except Exception as e:
                self._log(f"Local camera scan error: {e}")
        else:
            # fallback brute force small range
            for i in range(4):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    cam_list.append(i)
                cap.release()

        for c in cam_list:
            self.add_camera_source(c)
        self._log(f"Found {len(cam_list)} local camera(s).")

        # network scan in background (non-blocking)
        def net_scan():
            self._log("Scanning network cameras (this may take a while)...")
            try:
                net = camconnect.scan_network_cameras() if camconnect else []
                for ip in net:
                    url = f"rtsp://{ip}:554"
                    self.add_camera_source(url)
                self._log(f"Network scan found {len(net)} cameras.")
            except Exception as e:
                self._log(f"Network scan error: {e}")
        threading.Thread(target=net_scan, daemon=True).start()

    def add_camera_source(self, source):
        """
        Add camera source to the dashboard. We create a small queue and a worker but
        do not immediately start the worker unless its page is visible.
        """
        if source in self.camera_sources:
            return
        self.camera_sources.append(source)
        label = f"CAM_{len(self.camera_sources)}"
        self.frame_queues[label] = queue.Queue(maxsize=4)  # small queue to avoid backlog
        # create worker instance (not started yet)
        self.camera_workers[label] = CameraWorker(source, self.frame_queues[label], label)
        self._update_page_labels()
        self._log(f"Added camera source [{label}]: {source}")

    # -----------------------
    # Page and worker management
    # -----------------------
    def _camera_labels_for_page(self, page_index):
        start = page_index * self.per_page
        labels = []
        for i in range(self.per_page):
            idx = start + i
            if idx < len(self.camera_sources):
                labels.append((idx, self.camera_sources[idx]))
            else:
                labels.append((None, None))
        return labels

    def _update_page_labels(self):
        pairs = self._camera_labels_for_page(self.current_page)
        for slot, (idx, src) in enumerate(pairs):
            if idx is None:
                self.cam_name_labels[slot].configure(text=f"Camera #{slot+1} (empty)")
            else:
                self.cam_name_labels[slot].configure(text=f"Camera #{idx+1}")

    def _start_worker_for_slot(self, slot_index, cam_index):
        """
        Ensure the worker thread for the camera is running. If a thread was stopped,
        create a new one (threads cannot be restarted).
        """
        if cam_index is None:
            self._set_image_on_label(slot_index, None, text="No Camera")
            return

        label = f"CAM_{cam_index+1}"
        worker = self.camera_workers.get(label)
        if worker is None:
            q = self.frame_queues.get(label) or queue.Queue(maxsize=4)
            self.frame_queues[label] = q
            worker = CameraWorker(self.camera_sources[cam_index], q, label)
            self.camera_workers[label] = worker

        if not worker.is_alive():
            try:
                worker.start()
                self._log(f"Started worker for {label}")
            except RuntimeError:
                # recreate if the old thread cannot be restarted
                w = CameraWorker(self.camera_sources[cam_index], self.frame_queues[label], label)
                self.camera_workers[label] = w
                w.start()
                self._log(f"Restarted worker for {label}")

    # -----------------------
    # Core UI update loop: read from queues and update displays
    # -----------------------
    def _update_frames_periodic(self):
        """
        Called periodically (via .after). For each visible camera slot:
          - ensure the worker is running
          - drain the queue keeping only the newest frame
          - check hazards for critical keywords and manage timers + beep
          - display the latest frame
        """
        pairs = self._camera_labels_for_page(self.current_page)
        for slot, (idx, src) in enumerate(pairs):
            if idx is not None:
                self._start_worker_for_slot(slot, idx)
                label = f"CAM_{idx+1}"
                q = self.frame_queues.get(label)
                latest = None
                if q:
                    # Drain queue, keep only newest (non-blocking)
                    try:
                        while True:
                            latest = q.get_nowait()
                    except Exception:
                        pass

                if latest:
                    cam_label, frame, hazards = latest
                    cam_number = idx + 1
                    now = time.time()
                    critical_seen = False

                    # For each hazard, if critical mark and update timer.
                    # Play beep only when first seen (is_new).
                    for h in (hazards or []):
                        if is_critical(h):
                            is_new = cam_number not in self.critical_cameras
                            self.critical_cameras.add(cam_number)
                            self.critical_timers[cam_number] = now
                            critical_seen = True
                            if is_new:
                                # Play beep on new critical event
                                try:
                                    _beep()
                                except Exception:
                                    pass
                            self._log(f"Critical hazard on Camera #{cam_number}: {h}")

                    # If no critical hazard in this frame, check for timeout to clear
                    if not critical_seen and cam_number in self.critical_timers:
                        if now - self.critical_timers[cam_number] > self.critical_timeout:
                            # Clear camera from critical lists
                            self.critical_cameras.discard(cam_number)
                            try:
                                del self.critical_timers[cam_number]
                            except Exception:
                                pass
                            self._log(f"Camera #{cam_number} cleared from critical list after timeout.")

                    # Display the frame (BGR)
                    self._set_image_on_label(slot, frame)
                else:
                    # No latest frame available (worker may not have produced frames yet)
                    pass
            else:
                # Empty slot
                self._set_image_on_label(slot, None, text="No Camera")

        # Schedule next invocation
        self.after(80, self._update_frames_periodic)

    # -----------------------
    # Helper: convert BGR frame to Tk image and display
    # -----------------------
    def _set_image_on_label(self, slot_index, bgr_frame, text=None):
        lbl = self.image_labels[slot_index]
        if bgr_frame is None:
            lbl.configure(text=text or "No feed", image=None)
            return
        try:
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            w = lbl.winfo_width() or 400
            h = lbl.winfo_height() or 300
            pil.thumbnail((w, h))
            photo = ImageTk.PhotoImage(pil)
            # keep reference to avoid GC
            lbl.image = photo
            lbl.configure(image=photo, text="")
        except Exception as e:
            lbl.configure(text=f"Render error: {e}", image=None)

    # -----------------------
    # Paging controls
    # -----------------------
    def _next_page(self):
        max_pages = math.ceil(len(self.camera_sources) / self.per_page) if self.camera_sources else 1
        if self.current_page + 1 < max_pages:
            self.current_page += 1
            self._update_page_labels()
        else:
            self._log("Already at last page.")

    def _prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self._update_page_labels()
        else:
            self._log("Already at first page.")

    # -----------------------
    # Jump to specific camera
    # -----------------------
    def _jump_popup(self):
        try:
            ans = simpledialog.askinteger("Jump to Camera", "Enter camera number (1-based):", parent=self, minvalue=1)
            if ans is None:
                return
            cam_num = int(ans)
            if cam_num < 1 or cam_num > len(self.camera_sources):
                messagebox.showerror("Invalid", "Camera number out of range.")
                return
            zero_index = cam_num - 1
            page = zero_index // self.per_page
            self.current_page = page
            self._update_page_labels()
            self._log(f"Jumped to Camera #{cam_num} (page {page+1})")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # -----------------------
    # Critical cameras popup (with Refresh)
    # -----------------------
    def _show_critical_list(self):
        win = ctk.CTkToplevel(self)
        win.title("Critical Cameras")
        win.geometry("320x420")
        win.transient(self)

        lbl = ctk.CTkLabel(win, text="Cameras with recent critical hazards:")
        lbl.pack(padx=8, pady=8)

        listbox = ctk.CTkTextbox(win, height=260)
        listbox.pack(padx=8, pady=8, fill="both", expand=True)

        def refresh_list():
            listbox.delete("1.0", "end")
            if not self.critical_cameras:
                listbox.insert("end", "No critical cameras detected.\n")
            else:
                for camno in sorted(self.critical_cameras):
                    listbox.insert("end", f"Camera #{camno}\n")

        refresh_list()

        btn_frame = ctk.CTkFrame(win)
        btn_frame.pack(pady=8)

        refresh_btn = ctk.CTkButton(btn_frame, text="Refresh", command=refresh_list)
        refresh_btn.pack(side="left", padx=6)

        show_btn = ctk.CTkButton(btn_frame, text="Show First Critical", command=lambda: self._show_first_critical(win))
        show_btn.pack(side="left", padx=6)

        win.focus()
        win.grab_set()

    def _show_first_critical(self, win):
        if not self.critical_cameras:
            messagebox.showinfo("Info", "No critical cameras.")
            return
        first = next(iter(self.critical_cameras))
        page = (first - 1) // self.per_page
        self.current_page = page
        self._update_page_labels()
        self._log(f"Showing critical Camera #{first}")
        try:
            win.destroy()
        except Exception:
            pass

    # -----------------------
    # Shutdown: stop workers and exit
    # -----------------------
    def on_closing(self):
        for w in list(self.camera_workers.values()):
            try:
                w.stop()
            except Exception:
                pass
        self.destroy()

# -----------------------
# Run the app
# -----------------------
if __name__ == "__main__":
    app = RoadHazardApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
