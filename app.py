from flask import Flask, Response, jsonify
import cv2
import torch

app = Flask(__name__)

# Загрузка модели YOLO
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', source='github')

# Настройка камеры
camera = cv2.VideoCapture(0)  # 0 — первая веб-камера; замените на 1, если подключена вторая

def generate_frames():
    while True:
        success, frame = camera.read()  # Считывание кадра с камеры
        if not success:
            break

        # Инференс с использованием YOLO
        results = model(frame)
        annotated_frame = results.render()[0]  # Получаем кадр с аннотациями

        # Конвертируем изображение в формат JPEG
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        # Генерация фрейма для Flask
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/")
def index():
    return "YOLO Webcam API is running!"

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')