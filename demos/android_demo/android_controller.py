import os

import matplotlib.pyplot as plt
import numpy as np
import cv2
from flask import Flask, jsonify, request, send_file
from gradio_client import Client
from android_service import *

# 导入多线程控制模块
app = Flask(__name__)

# 设置保存图片的目录
UPLOAD_FOLDER = os.getcwd()  # 获取当前工作目录
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 定义队列
queue = None
feedback_q = None
RES = 256
file_path = ""


@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'Connection success!Model: ScribblePrompt-main'}), 200


@app.route('/load_img', methods=['POST'])
def loadImg():
    global file_path
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        file_extension = os.path.splitext(file.filename)[1]  # 获取文件扩展名
        filename = 'tempImg' + file_extension  # 设置文件名为 tempImg 后加上文件扩展名
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # 拼接文件的完整路径
        file.save(file_path)  # 保存文件

        img = cv2.imread(file_path, cv2.COLOR_BGR2RGB)
        if img is not None:
            feedback = change_input_img(img)
            print(f"feedback: {feedback}")
            init_current_label()
            print('current label is reset to 1.')

            return jsonify({"message": "File successfully uploaded", "file": filename}), 200
        else:
            return jsonify({"error": "Image-loading failed"}), 500


@app.route('/switch_label', methods=['GET'])
def switch_label():
    feedback = change_current_label()
    print(f"feedback: current label is {feedback}")

    return jsonify({"message": f"label is changed to {feedback}"}), 200


@app.route('/click', methods=['POST'])
def get_click():
    data = request.json
    x = int(data.get("x", -1))
    y = int(data.get("y", -1))
    flag = data.get("flag", 1)
    # print("x = ", x, ", y = ", y, ", flag = ", flag)
    feedback = click_seg([x, y])
    print(f"feedback: {feedback}")

    return jsonify({"message": f"Received data - X: {x}, Y: {y}, Flag: {flag}"}), 200


@app.route('/paint', methods=['POST'])
def load_painting():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        file_extension = os.path.splitext(file.filename)[1]  # 获取文件扩展名
        filename = 'tempPaint' + file_extension  # 设置文件名为 tempPaint 后加上文件扩展名
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # 拼接文件的完整路径
        file.save(filepath)  # 保存文件

        painting = cv2.imread(filepath, cv2.COLOR_BGR2RGB)
        if painting is not None:
            feedback = paint_seg(painting)
            # print(f"feedback: {feedback}")

            return jsonify({"message": "File successfully uploaded"}), 200
        else:
            return jsonify({"error": "Image-loading failed"}), 500


@app.route('/get_image', methods=['GET'])
def get_image():
    file_path = os.path.join(os.getcwd(), 'tempImg.jpg')
    mimetype = 'image/jpg'
    if not os.path.exists(file_path):
        file_path = os.path.join(os.getcwd(), 'tempImg.png')
        mimetype = 'image/png'
    try:
        return send_file(file_path, mimetype=mimetype), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# @app.route('/undo', methods=['GET'])
# def undo():
#
#     print(f"feedback: {feedback}")
#
#     try:
#         return jsonify({"feedback": feedback}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


@app.route('/finish', methods=['GET'])
def finish():
    feedback = get_output()
    # print(f"feedback: {feedback}")
    cv2.imwrite(file_path, feedback)

    try:
        return jsonify({"message": "success"}), 200
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # 启动 GUI 并获取控制队列
    # queue, feedback_q = run_gui_in_thread()
    #
    # # 等待 GUI 启动
    # feedback = feedback_q.get()  # 获取从 GUI 线程返回的反馈信息
    # print(feedback)
    # time.sleep(1)

    app.run(debug=False, host='0.0.0.0', port=5000)
