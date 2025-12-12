import json
import subprocess
import threading
import time
import uuid
from collections import deque

import cv2
import yaml
from flask import Flask, Response


class ClientManager:
    def __init__(self, max_length=100):
        self.clients = {}
        self.condition = threading.Condition()
        self.max_length = max_length

    def add_client(self):
        with self.condition:
            client_id = str(uuid.uuid4())
            self.clients[client_id] = {
                "json_queue": deque(maxlen=self.max_length),
                "frame_queue": deque(maxlen=self.max_length)
            }
            print("new client: ", client_id)
            print(
                "current clients [{}]:\n {}".format(
                    len(self.clients), self.clients.keys()
                )
            )
            return client_id

    def remove_client(self, client_id):
        with self.condition:
            if client_id in self.clients:
                del self.clients[client_id]
                print("close client: ", client_id)
                print(
                    "current clients [{}]:\n {}".format(
                        len(self.clients), self.clients.keys()
                    )
                )

    def update_json(self, json_data):
        with self.condition:
            for client in self.clients.values():
                client["json_queue"].append(json_data)
            self.condition.notify_all()

    def update_frame(self, frame_data):
        with self.condition:
            for client in self.clients.values():
                client["frame_queue"].append(frame_data)
            self.condition.notify_all()

    def get_json(self, client_id):
        with self.condition:
            while not self.clients[client_id]["json_queue"]:
                self.condition.wait()
            return self.clients[client_id]["json_queue"].popleft()

    def get_frame(self, client_id):
        with self.condition:
            while not self.clients[client_id]["frame_queue"]:
                self.condition.wait()
            return self.clients[client_id]["frame_queue"].popleft()


client_manager = ClientManager()
app_json = Flask("json_server")
app_mjpeg = Flask("mjpeg_server")


@app_json.route("/")
def json_endpoint():
    client_id = client_manager.add_client()

    def generate_json(client_id):
        try:
            while True:
                json_data = client_manager.get_json(client_id)
                yield json_data
        finally:
            client_manager.remove_client(client_id)

    return Response(generate_json(client_id), content_type="text/event-stream; charset=utf-8")


@app_mjpeg.route("/")
def mjpeg_endpoint():
    client_id = client_manager.add_client()

    def generate_mjpeg(client_id):
        try:
            while True:
                frame_data = client_manager.get_frame(client_id)
                yield frame_data
        finally:
            client_manager.remove_client(client_id)

    response = Response(generate_mjpeg(
        client_id), content_type="multipart/x-mixed-replace; boundary=FRAME")
    return response


def replace_yaml_field(filename, field, value):
    try:
        with open(filename, "r") as file:
            config = yaml.safe_load(file)
            config[field] = value
        with open(filename, "w") as file:
            yaml.safe_dump(config, file)
    except FileNotFoundError:
        print(f"{filename} not found.")


def start_json_server(json_deque, json_port):
    def json_server_thread():
        while True:
            try:
                json_data = json_deque.popleft()
                json_data_str = json.dumps(
                    json_data, ensure_ascii=False) + "\n"
                json_data_bytes = json_data_str.encode("UTF-8")
                client_manager.update_json(json_data_bytes)
            except IndexError:
                time.sleep(0.001)

    threading.Thread(target=json_server_thread, daemon=True).start()
    threading.Thread(
        target=app_json.run, kwargs={"host": "0.0.0.0", "port": json_port}, daemon=True
    ).start()


def start_mjpeg_server(frame_deque, mjpeg_port, size=None, quality=50):
    def mjpeg_server_thread():
        while True:
            try:
                frame = frame_deque.popleft()
                if size:
                    frame = cv2.resize(frame, size)
                _, jpg_str = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                frame_data = (
                    b"--FRAME\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: "
                    + str(len(jpg_str)).encode()
                    + b"\r\n\r\n"
                    + jpg_str.tobytes()
                    + b"\r\n"
                )
                client_manager.update_frame(frame_data)
            except IndexError:
                time.sleep(0.001)

    threading.Thread(target=mjpeg_server_thread, daemon=True).start()
    threading.Thread(
        target=app_mjpeg.run, kwargs={"host": "0.0.0.0", "port": mjpeg_port}, daemon=True
    ).start()


def create_mediamtx_process(rtsp_port):
    mediamtx_executable = "../mediamtx/mediamtx"
    mediamtx_config = "../mediamtx/mediamtx.yml"
    replace_yaml_field(mediamtx_config, "rtspAddress", f":{rtsp_port}")
    mediamtx_process = subprocess.Popen(
        [mediamtx_executable, mediamtx_config], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return mediamtx_process


def create_ffmpeg_process(rtsp_port, fps, orig_size, out_size, bitrate):
    ffmpeg_command = [
        "ffmpeg",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", "{}x{}".format(*orig_size),  # 原始图像尺寸
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-b:v", "{}k".format(bitrate),  # 视频比特率
        "-maxrate", "{}k".format(bitrate),  # 最大比特率
        "-g", str(2 * fps),
        "-max_delay", "1000000",  # 最大延迟1000ms
        "-bufsize", "{}k".format(2 * bitrate),  # 缓冲区大小
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        "rtsp://127.0.0.1:{}/rtsp".format(rtsp_port),
    ]

    if out_size is not None:
        ffmpeg_command.insert(11, "-vf")
        ffmpeg_command.insert(12, "scale={}:{}".format(*out_size))

    ffmpeg_process = subprocess.Popen(
        ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return ffmpeg_process


def process_video(frame_deque, rtsp_port, out_size, fps, bitrate, timeout):
    while not frame_deque:
        time.sleep(0.001)
    frame = frame_deque.popleft()
    height, width, _ = frame.shape
    mediamtx_process = create_mediamtx_process(rtsp_port)
    ffmpeg_process = create_ffmpeg_process(
        rtsp_port, fps, (width, height), out_size, bitrate)
    check_ffmpeg_time = time.time()
    last_data_time = time.time()

    try:
        while True:
            if time.time() - check_ffmpeg_time >= 5:
                check_ffmpeg_time = time.time()
                if ffmpeg_process.poll() is not None:
                    print("ffmpeg process terminated, restarting...")
                    ffmpeg_process.stdin.close()
                    ffmpeg_process.terminate()
                    ffmpeg_process = create_ffmpeg_process(
                        rtsp_port, fps, (width, height), out_size, bitrate)

            try:
                frame = frame_deque.popleft()
                ffmpeg_process.stdin.write(frame.tobytes())
                ffmpeg_process.stdin.flush()
                last_data_time = time.time()
            except IndexError:
                if time.time() - last_data_time > timeout:
                    print(
                        "No new data received for specified timeout period. Exiting...")
                    break
                time.sleep(0.001)
            except BrokenPipeError as e:
                print(f"BrokenPipeError: {e}")

    finally:
        ffmpeg_process.stdin.close()
        mediamtx_process.terminate()
        ffmpeg_process.terminate()


def start(data_deque, json_port, rtsp_port=None, mjpeg_port=None, quality=50, out_size=None, fps=25, bitrate=2000, max_length=100, timeout=10):
    frame_deque = deque(maxlen=max_length)
    json_deque = deque(maxlen=max_length)

    def split_data():
        while True:
            try:
                data = data_deque.popleft()
                json_deque.append(data[0])
                frame_deque.append(data[1])
            except IndexError:
                time.sleep(0.001)

    threading.Thread(target=split_data, daemon=True).start()
    if mjpeg_port:
        start_mjpeg_server(frame_deque, mjpeg_port, out_size, quality)
    else:
        threading.Thread(
            target=process_video,
            args=(frame_deque, rtsp_port, out_size, fps, bitrate, timeout),
            daemon=True,
        ).start()
    start_json_server(json_deque, json_port)

    print("json url: http://127.0.0.1:{}".format(json_port))
    if mjpeg_port:
        print("mjpeg url: http://127.0.0.1:{}".format(mjpeg_port))
    else:
        print("rtsp url: rtsp://127.0.0.1:{}/rtsp".format(rtsp_port))


if __name__ == "__main__":
    import argparse
    import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=str,
                        help="video file location")
    parser.add_argument("--json_port", type=int, help="json port")
    parser.add_argument("--rtsp_port", type=int, help="rtsp port ")
    parser.add_argument("--mjpeg_port", type=int, help="mjpeg_port")
    args = parser.parse_args()

    data_deque = deque(maxlen=100)

    def read_video(video_path, data_deque):
        frame_id = 1
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            time.sleep(0.01)
            if not ret:
                break
            json_data = {
                "frame_id": frame_id,
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            frame_id += 1
            data_deque.append([json_data, frame])
        cap.release()

    start(data_deque, args.json_port, args.rtsp_port)
    read_video(args.video, data_deque)
