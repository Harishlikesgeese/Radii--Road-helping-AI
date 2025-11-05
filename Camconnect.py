import cv2
import socket
import threading
from Video_and_Image_Anylyzer import live_detection  # import from your analyzer script


def check_local_cameras(max_index=10):
    """Search for locally connected USB/IP cameras."""
    available = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"[✔] Local camera found at index {index}")
            available.append(index)
        cap.release()
    return available


def scan_network_cameras(port=554, timeout=0.2):
    """
    Discover IP cameras on the local subnet via RTSP (port 554).
    For larger networks, consider narrowing the subnet range.
    """
    local_ip = socket.gethostbyname(socket.gethostname())
    subnet = ".".join(local_ip.split(".")[:-1]) + "."
    found_cameras = []
    threads = []

    def ping_ip(ip):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        try:
            s.connect((ip, port))
            print(f"[✔] Possible IP camera found: rtsp://{ip}:{port}")
            found_cameras.append(ip)
        except:
            pass
        finally:
            s.close()

    for i in range(1, 255):
        ip = subnet + str(i)
        t = threading.Thread(target=ping_ip, args=(ip,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    return found_cameras


def run_camera_feed(camera_source, window_name):
    """
    Run YOLO live detection on a camera source.
    Each camera runs in a separate thread.
    """
    try:
        print(f"🎥 Starting detection on {camera_source}")
        live_detection(camera_source)
    except Exception as e:
        print(f"[❌] Error with {camera_source}: {e}")


def main():
    print("🔍 Searching for available cameras...")
    local_cams = check_local_cameras()
    net_cams = scan_network_cameras()

    camera_sources = local_cams + [f"rtsp://{ip}:554" for ip in net_cams]

    if not camera_sources:
        print("❌ No cameras detected on the network or locally.")
        return

    print(f"\n🎬 Detected {len(camera_sources)} camera(s). Starting multi-feed analysis...")

    threads = []
    for i, cam in enumerate(camera_sources):
        window_name = f"Camera Feed {i+1}"
        t = threading.Thread(target=run_camera_feed, args=(cam, window_name))
        t.daemon = True
        t.start()
        threads.append(t)

    # Keep main thread alive until user quits all feeds
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\n🛑 Stopping all feeds...")


if __name__ == "__main__":
    main()
