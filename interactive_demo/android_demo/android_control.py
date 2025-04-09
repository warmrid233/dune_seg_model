import argparse
import time

import cv2
import torch

from isegm.utils import exp
from android_model import run_gui_in_thread

from flask import Flask, jsonify, request, send_file
import os

connect = Flask(__name__)

# 设置保存图片的目录
UPLOAD_FOLDER = os.getcwd()  # 获取当前工作目录
connect.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='The path to the checkpoint. '
                             'This can be a relative path (relative to cfg.INTERACTIVE_MODELS_PATH) '
                             'or an absolute path. The file extension can be omitted.')

    parser.add_argument('--gpu', type=int, default=0,
                        help='Id of GPU to use.')

    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Use only CPU for inference.')

    parser.add_argument('--limit-longest-size', type=int, default=800,
                        help='If the largest side of an image exceeds this value, '
                             'it is resized so that its largest side is equal to this value.')

    parser.add_argument('--cfg', type=str, default="config.yml",
                        help='The path to the config file.')

    parser.add_argument('--eval-ritm', action='store_true', default=False)
    args = parser.parse_args()

    if args.cpu:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f'cuda:{args.gpu}')
    cfg = exp.load_config_file(args.cfg, return_edict=True)

    return args, cfg


# 与android端连接测试路由
@connect.route('/test', methods=['GET'])
def test():
    return jsonify({'message': '模型连接成功!当前模型：simpleCLick1.0'}), 200


# 接收上传图片路由
@connect.route('/load_img', methods=['POST'])
def loadImg():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # 固定文件名为 tempImg，扩展名根据原文件类型决定
        file_extension = os.path.splitext(file.filename)[1]  # 获取文件扩展名
        filename = 'tempImg' + file_extension  # 设置文件名为 tempImg 后加上文件扩展名

        # 拼接文件的完整路径
        filepath = os.path.join(connect.config['UPLOAD_FOLDER'], filename)

        # 保存文件
        file.save(filepath)

        img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
        if img is not None:
            queue.put(("load_img", img))

        feedback = feedback_q.get()  # 获取从 GUI 线程返回的反馈信息
        print(feedback)

        return jsonify({"message": "File successfully uploaded", "file": filename}), 200


# 点击交互路由
@connect.route('/click', methods=['POST'])
def get_click():
    data = request.json
    x = data.get("x", -1)
    y = data.get("y", -1)
    flag = data.get("flag", 1)
    print("x = ", x, ", y = ", y, ", flag = ", flag)

    if flag == 0:
        queue.put(("get_click", x, y, False))
    else:
        queue.put(("get_click", x, y, True))

    feedback = feedback_q.get()  # 获取从 GUI 线程返回的反馈信息
    print(feedback)

    return jsonify({"message": f"Received data - X: {x}, Y: {y}, Flag: {flag}"})


# android获取图片路由
@connect.route('/get_image', methods=['GET'])
def get_image():
    # 设置图片的文件路径
    image_path = os.path.join(os.getcwd(), 'tempImg.jpg')
    mimetype = 'image/jpg'
    # 如果是 PNG 文件，则使用 'image/png'
    if not os.path.exists(image_path):
        image_path = os.path.join(os.getcwd(), 'tempImg.png')
        mimetype = 'image/png'
    try:
        # 发送图片文件
        return send_file(image_path, mimetype=mimetype)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 撤回操作路由
@connect.route('/undo', methods=['GET'])
def undo():
    queue.put(('undo',))
    feedback = feedback_q.get()  # 获取从 GUI 线程返回的反馈信息
    print(feedback)

    try:
        # 发送图片文件
        return jsonify({"feedback": feedback}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 交互完成路由
@connect.route('/finish', methods=['GET'])
def finish():
    queue.put(('finish',))
    feedback = feedback_q.get()  # 获取从 GUI 线程返回的反馈信息
    print(feedback)

    try:
        # 发送图片文件
        return jsonify({"feedback": feedback}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


queue = None
feedback_q = None
if __name__ == "__main__":
    # 解析命令行参数
    args, cfg = parse_args()

    # 启动 GUI 并获取控制队列
    queue, feedback_q = run_gui_in_thread(args, cfg)

    # 等待 GUI 启动
    feedback = feedback_q.get()  # 获取从 GUI 线程返回的反馈信息
    print(feedback)
    time.sleep(1)
    connect.run(debug=False, host='0.0.0.0', port=5000)

    # while True:
    #     action = input("请输入操作命令(loadImg, click, undo, reset, finish)" + '\n')
    #
    #     # 通过队列向 GUI 发送命令
    #     if action == 'loadImg':
    #         img = cv2.cvtColor(cv2.imread("tempImg.jpg"), cv2.COLOR_BGR2RGB)
    #         if img is None:
    #             img = cv2.cvtColor(cv2.imread("tempImg.jpg"), cv2.COLOR_BGR2RGB)
    #         queue.put(("load_img", img))
    #
    #     elif action == 'click':
    #         x = int(input("输入x的值："))
    #         y = int(input("输入y的值："))
    #         flag = input("is positive(default)? (1 - positive / 0 - negative)" + '\n')
    #         if flag == '0':
    #             queue.put(("get_click", x, y, False))
    #         else:
    #             queue.put(("get_click", x, y, True))
    #
    #     elif action == 'undo':
    #         queue.put(("undo",))
    #
    #     elif action == 'reset':
    #         queue.put(("reset",))
    #
    #     elif action == 'finish':
    #         queue.put(("finish",))
    #
    #     elif action == 'exit':
    #         queue.put(("exit",))
    #
    #     else:
    #         continue
    #
    #     # 如果需要从 GUI 线程获取反馈信息
    #     feedback = feedback_q.get()  # 获取从 GUI 线程返回的反馈信息
    #     print(feedback)
