from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
from stream_handler import IntegratedStreamHandler
import cv2
import time
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'cv_safety_sys_secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

handler = None

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    global handler
    if handler is None or not handler.is_running:
        return
    
    while handler.is_running:
        frame, status = handler.get_frame()
        if frame is None:
            break
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        socketio.emit('data_update', status, broadcast=True)
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start', methods=['POST'])
def start_detection():
    global handler
    try:
        data = request.json or {}
        source_type = data.get('source_type', 'camera')
        
        if source_type == 'camera':
            video_source = data.get('camera_id', 0)
            try:
                video_source = int(video_source)
            except:
                video_source = 0
        elif source_type == 'video':
            video_source = data.get('video_path', '')
            if not video_source:
                return jsonify({'status': 'error', 'message': '请指定视频文件路径'}), 400
        else:
            video_source = 0
        
        pose_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                      'WebcamPoseDetection', 'models', 'pose_landmarker_full.task')
        handler = IntegratedStreamHandler(video_source=video_source, pose_model_path=pose_model_path)
        handler.start_capture()
        
        source_info = f"视频文件: {video_source}" if isinstance(video_source, str) else f"摄像头 {video_source}"
        return jsonify({'status': 'success', 'message': f'检测已启动 ({source_info})'})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/stop', methods=['POST'])
def stop_detection():
    global handler
    if handler:
        handler.stop_capture()
        handler = None
    return jsonify({'status': 'success', 'message': '检测已停止'})

@app.route('/api/toggle_selection', methods=['POST'])
def toggle_selection():
    global handler
    if not handler:
        return jsonify({'status': 'error', 'message': '检测未启动'}), 400
    
    data = request.json
    track_id = data.get('track_id')
    if track_id is None:
        return jsonify({'status': 'error', 'message': '缺少 track_id'}), 400
    
    selected = handler.toggle_selection(int(track_id))
    return jsonify({'status': 'success', 'selected': selected, 'track_id': track_id})

@app.route('/api/clear_selection', methods=['POST'])
def clear_selection():
    global handler
    if handler:
        handler.clear_selection()
    return jsonify({'status': 'success', 'message': '已清除所有选择'})

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    global handler
    if not handler:
        return jsonify({'alerts': []})
    
    alerts = handler.get_alerts()
    return jsonify({'alerts': alerts})

@socketio.on('connect')
def handle_connect():
    emit('status', {'message': '已连接到服务器'})

@socketio.on('disconnect')
def handle_disconnect():
    pass

@socketio.on('click_video')
def handle_video_click(data):
    global handler
    if not handler:
        return
    
    x = data.get('x')
    y = data.get('y')
    
    for detection in handler.relic_detections:
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        if x1 <= x <= x2 and y1 <= y <= y2:
            track_id = detection.get('track_id')
            if track_id is not None:
                selected = handler.toggle_selection(track_id)
                emit('selection_changed', {
                    'track_id': track_id, 
                    'selected': selected
                }, broadcast=True)
                break

if __name__ == '__main__':
    print("Test WebUI - http://localhost:1145")
    socketio.run(app, host='0.0.0.0', port=1145, debug=False)
