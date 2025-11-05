# RoadHazard_Dashboard.py
import os
import threading
import time
import math
import queue
import cv2
import customtkinter as ctk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk

# --- Robust import of user's analyzer module (tries multiple filenames) ---
analyzer = None
analyzer_module_names = [
    "Video_and_Image_Anylyzer",       # snake_case import used by Camconnect
    "Video and Image Anylyzer",       # spaced filename possibility
    "Video and Image Anylyzer.py",    # fallback path
    "Video_and_Image_Analyzer"        # alternative spelling
]
for name in analyzer_module_names:
    try:
        # Try direct import first
        analyzer = __import__(name)
        break
    except Exception:
        pass

# If not imported yet, attempt to load from file 'Video and Image Anylyzer.py' in cwd
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

# Final fallback: try 'Video_and_Image_Anylyzer.py' filename
if analyzer is None and os.path.exists(os.path.join(os.getcwd(), "Video_and_Image_Anylyzer.py")):
    import importlib.util
    spec = importlib.util.spec_from_file_location("analyzer_custom2", os.path.join(os.getcwd(), "Video_and_Image_Anylyzer.py"))
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)

# --- Import Camconnect utilities if available ---
camconnect = None
try:
    import Camconnect as camconnect
except Exception:
    # try alternative filename
    cam_path = os.path.join(os.getcwd(), "Camconnect.py")
    if os.path.exists(cam_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("camconnect_custom", cam_path)
        camconnect = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(camconnect)

# Validate we have model access
model = None
ROAD_HAZARDS = {}
FIX_MAP = {}
if analyzer is not None:
    # attempt to access model and hazard maps from analyzer module
    model = getattr(analyzer, "model", None)
    ROAD_HAZARDS = getattr(analyzer, "ROAD_HAZARDS", {})
    FIX_MAP = getattr(analyzer, "FIX_MAP", {})

# If no model found, we cannot overlay YOLO boxes; notify user at start
MODEL_AVAILABLE = model is not None

# --- Helper to classify critical hazards ---
CRITICAL_KEYWORDS = ["crash", "accident", "🚨", "fire", "🔥", "smoke", "⚠️"]
def is_critical(hazard_text: str):
    if not hazard_text:
        return False
    low = hazard_text.lower()
    for kw in CRITICAL_KEYWORDS:
        if kw in low or kw in hazard_text:
            return True
    return False

# --- Camera runner per source (captures frames and runs detection if model available) ---
class CameraWorker(threading.Thread):
    def __init__(self, source, frame_queue, camera_id_label):
        super().__init__(daemon=True)
        self.source = source
        self.frame_queue = frame_queue  # queue for GUI frames
        self.camera_id_label = camera_id_label
        self._stop = threading.Event()
        self.cap = None

    def stop(self):
        self._stop.set()
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass

    def run(self):
        try:
            # Open capture (source may be int index or rtsp URL)
            try:
                src = int(self.source)
            except Exception:
                src = self.source
            self.cap = cv2.VideoCapture(src)
            if not self.cap.isOpened():
                # try with small delay and reopen
                time.sleep(0.2)
                if not self.cap.isOpened():
                    self.frame_queue.put((self.camera_id_label, None, f"[Error] Cannot open {self.source}"))
                    return

            while not self._stop.is_set():
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    # small pause and continue
                    time.sleep(0.05)
                    continue

                overlay_frame = frame.copy()
                hazards_found = []

                if MODEL_AVAILABLE:
                    try:
                        # Use analyzer's model predict on frame
                        results = model.predict(source=frame, conf=analyzer.SCORE_THRESH if hasattr(analyzer, "SCORE_THRESH") else 0.35, verbose=False)
                        for r in results:
                            # r.boxes may be empty; iterate safely
                            if hasattr(r, "boxes") and r.boxes is not None:
                                for box_obj in r.boxes:
                                    # attempt to get class id and label
                                    try:
                                        cls_id = int(box_obj.cls)
                                        label = r.names[cls_id]
                                    except Exception:
                                        # fallback: try reading from box_obj.cls CPU numpy
                                        try:
                                            cls_id = int(box_obj.cls.cpu().numpy())
                                            label = r.names[cls_id]
                                        except Exception:
                                            continue

                                    # bounding box coordinates
                                    try:
                                        xy = box_obj.xyxy[0]
                                        x1, y1, x2, y2 = map(int, xy)
                                    except Exception:
                                        continue

                                    # draw rectangle and label
                                    cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    label_text = f"{label}"
                                    cv2.putText(overlay_frame, label_text, (x1, y1 - 6),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                                    # record hazard if in ROAD_HAZARDS
                                    if label in ROAD_HAZARDS:
                                        hazards_found.append(ROAD_HAZARDS[label])

                    except Exception as e:
                        # If model prediction fails, push the raw frame with an annotation message
                        cv2.putText(overlay_frame, f"Model error: {e}", (10, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

                # Put into queue for GUI to update. Convert BGR->RGB now for faster conversion later.
                self.frame_queue.put((self.camera_id_label, overlay_frame, hazards_found))
                # pace the loop to avoid excessive CPU used by predict (YOLO will throttle itself)
                time.sleep(0.02)
        finally:
            try:
                if self.cap:
                    self.cap.release()
            except Exception:
                pass

# --- Main GUI ---
class RoadHazardApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Road Hazard Dashboard")
        self.geometry("1200x700")
        ctk.set_appearance_mode("Light")
        ctk.set_default_color_theme("blue")

        # state
        self.camera_sources = []  # list of sources (ints or rtsp strings)
        self.camera_workers = {}  # source -> CameraWorker
        self.frame_queues = {}    # source_label -> Queue
        self.current_page = 0     # page index, 2 cameras per page
        self.per_page = 2

        # critical set (camera indices that reported critical hazard recently)
        self.critical_cameras = set()

        # layout
        self._build_layout()

        # start discovery of local cameras in background
        threading.Thread(target=self._initial_discover, daemon=True).start()

        # periodic GUI frame updater
        self.after(50, self._update_frames_periodic)

    def _build_layout(self):
        # Top controls
        top_frame = ctk.CTkFrame(self, height=60)
        top_frame.pack(side="top", fill="x", padx=6, pady=6)

        self.jump_btn = ctk.CTkButton(top_frame, text="Jump to Camera", command=self._jump_popup)
        self.jump_btn.pack(side="left", padx=(10,6))

        self.crit_btn = ctk.CTkButton(top_frame, text="Critical Cameras", command=self._show_critical_list)
        self.crit_btn.pack(side="left")

        # center main area
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(side="top", fill="both", expand=True, padx=6, pady=(0,6))

        # left and right video panels within main_frame
        panels_frame = ctk.CTkFrame(main_frame)
        panels_frame.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        # two equal frames (camera views)
        self.cam_frames = []
        self.image_labels = []
        self.cam_name_labels = []

        for i in range(2):
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

        # Right-edge gray arrow for paging
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

    def _log(self, text):
        ts = time.strftime("%H:%M:%S")
        self.log_box.insert("end", f"[{ts}] {text}\n")
        self.log_box.see("end")

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
        # add them to sources
        for c in cam_list:
            self.add_camera_source(c)
        self._log(f"Found {len(cam_list)} local camera(s).")

        # start network scan in background (non-blocking)
        def net_scan():
            self._log("Scanning network cameras (this may take a while)...")
            try:
                if camconnect:
                    net = camconnect.scan_network_cameras()
                else:
                    net = []
                # transform ips to rtsp urls
                for ip in net:
                    url = f"rtsp://{ip}:554"
                    self.add_camera_source(url)
                self._log(f"Network scan found {len(net)} cameras.")
            except Exception as e:
                self._log(f"Network scan error: {e}")
        threading.Thread(target=net_scan, daemon=True).start()

    def add_camera_source(self, source):
        # Append to camera_sources if not duplicate
        if source in self.camera_sources:
            return
        self.camera_sources.append(source)
        # prepare frame queue and worker but don't start all workers now; start on demand when shown
        label = f"CAM_{len(self.camera_sources)}"
        q = queue.Queue(maxsize=4)
        self.frame_queues[label] = q

        # create worker but do not start until needed
        worker = CameraWorker(source, q, label)
        self.camera_workers[label] = worker

        self._log(f"Added camera source [{label}]: {source}")

        # If this addition makes current page not empty, refresh display
        self._update_page_labels()

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
        # Update camera name labels based on current page
        pairs = self._camera_labels_for_page(self.current_page)
        for slot, (idx, src) in enumerate(pairs):
            if idx is None:
                self.cam_name_labels[slot].configure(text=f"Camera #{slot+1} (empty)")
            else:
                self.cam_name_labels[slot].configure(text=f"Camera #{idx+1}")

    def _start_worker_for_slot(self, slot_index, cam_index):
        # cam_index is index in camera_sources
        if cam_index is None:
            # show blank image
            self._set_image_on_label(slot_index, None, text="No Camera")
            return

        # Identify label name
        label = f"CAM_{cam_index+1}"
        worker = self.camera_workers.get(label)
        if worker is None:
            # create and store
            q = self.frame_queues.get(label) or queue.Queue(maxsize=4)
            self.frame_queues[label] = q
            worker = CameraWorker(self.camera_sources[cam_index], q, label)
            self.camera_workers[label] = worker

        # start thread if not alive
        if not worker.is_alive():
            try:
                worker.start()
                self._log(f"Started worker for {label}")
            except RuntimeError:
                # threads cannot be restarted if previously stopped; recreate
                w = CameraWorker(self.camera_sources[cam_index], self.frame_queues[label], label)
                self.camera_workers[label] = w
                w.start()
                self._log(f"Restarted worker for {label}")

    def _update_frames_periodic(self):
        # For the two current slots, ensure workers are running and pull frames from queues
        pairs = self._camera_labels_for_page(self.current_page)
        for slot, (idx, src) in enumerate(pairs):
            if idx is not None:
                self._start_worker_for_slot(slot, idx)
                label = f"CAM_{idx+1}"
                q = self.frame_queues.get(label)
                if q:
                    try:
                        # non-blocking get latest frame (drain older frames)
                        latest = None
                        while True:
                            item = q.get_nowait()
                            latest = item
                    except Exception:
                        item = latest if 'latest' in locals() else None
                    if latest:
                        cam_label, frame, hazards = latest
                        # update critical cameras set if hazards include critical
                        for h in (hazards or []):
                            if is_critical(h):
                                self.critical_cameras.add(idx+1)  # camera numbers are 1-based
                                self._log(f"Critical hazard on Camera #{idx+1}: {h}")
                        # send to display
                        self._set_image_on_label(slot, frame)
                    else:
                        # no new frame, do nothing
                        pass
            else:
                # empty slot
                self._set_image_on_label(slot, None, text="No Camera")

        # schedule next update
        self.after(80, self._update_frames_periodic)

    def _set_image_on_label(self, slot_index, bgr_frame, text=None):
        lbl = self.image_labels[slot_index]
        if bgr_frame is None:
            # show placeholder
            lbl.configure(text=text or "No feed", image=None)
            return
        try:
            # Convert BGR to RGB and to PIL Image
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            # resize maintain aspect to fit in label
            w = lbl.winfo_width() or 400
            h = lbl.winfo_height() or 300
            pil.thumbnail((w, h))
            photo = ImageTk.PhotoImage(pil)
            # store reference to avoid garbage collection
            lbl.image = photo
            lbl.configure(image=photo, text="")
        except Exception as e:
            lbl.configure(text=f"Render error: {e}", image=None)

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

    def _jump_popup(self):
        # Ask for camera number (1-based)
        try:
            ans = simpledialog.askinteger("Jump to Camera", "Enter camera number (1-based):", parent=self, minvalue=1)
            if ans is None:
                return
            cam_num = int(ans)
            if cam_num < 1 or cam_num > len(self.camera_sources):
                messagebox.showerror("Invalid", "Camera number out of range.")
                return
            # compute page index that contains this camera
            zero_index = cam_num - 1
            page = zero_index // self.per_page
            self.current_page = page
            self._update_page_labels()
            self._log(f"Jumped to Camera #{cam_num} (page {page+1})")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _show_critical_list(self):
        # Show a simple window with camera numbers that are critical
        win = ctk.CTkToplevel(self)
        win.title("Critical Cameras")
        win.geometry("300x400")
        # make it transient to the main window and modal
        win.transient(self)

        lbl = ctk.CTkLabel(win, text="Cameras with recent critical hazards:")
        lbl.pack(padx=8, pady=8)
        listbox = ctk.CTkTextbox(win, height=260)
        listbox.pack(padx=8, pady=8, fill="both", expand=True)

        if not self.critical_cameras:
            listbox.insert("end", "No critical cameras detected recently.\n")
        else:
            for camno in sorted(self.critical_cameras):
                listbox.insert("end", f"Camera #{camno}\n")

        def on_select():
            try:
                if not self.critical_cameras:
                    messagebox.showinfo("Info", "No critical cameras to show.")
                    return
                # pick first and jump
                first = next(iter(self.critical_cameras))
                page = (first - 1) // self.per_page
                self.current_page = page
                self._update_page_labels()
                self._log(f"Showing critical Camera #{first}")
                win.destroy()
            except Exception as e:
                messagebox.showerror("Error", str(e))

        btn = ctk.CTkButton(win, text="Show First Critical", command=on_select)
        btn.pack(padx=8, pady=8)

        # Keep popup open until user closes it (modal)
        win.focus()
        win.grab_set()

    def on_closing(self):
        # stop all workers
        for w in list(self.camera_workers.values()):
            try:
                w.stop()
            except Exception:
                pass
        self.destroy()

if __name__ == "__main__":
    app = RoadHazardApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
