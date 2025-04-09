import cv2
import matplotlib

matplotlib.use('Agg')

import tkinter as tk

import torch

import threading
import queue

from isegm.inference import utils
from interactive_demo.app import InteractiveDemoApp


def start_gui(args, cfg, q=None, feedback_q=None):
    torch.backends.cudnn.deterministic = True
    checkpoint_path = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, args.checkpoint)
    model = utils.load_is_model(checkpoint_path, args.device, args.eval_ritm, cpu_dist_maps=True)

    root = tk.Tk()
    root.minsize(960, 480)
    app = InteractiveDemoApp(root, args, model)
    root.app = app

    # 线程处理消息队列
    def process_queue():
        while True:
            if q and not q.empty():
                command, *params = q.get()  # 命令和参数
                if command == "exit":
                    root.quit()
                    feedback_q.put("程序已退出")  # 向主线程返回反馈
                elif command == "load_img":
                    image = params[0]
                    app.load_img(image)
                    feedback_q.put("图片已加载")
                elif command == "get_click":
                    x = params[0]
                    y = params[1]
                    flag = params[2]
                    msg = app.test_click(flag, x, y)
                    if msg is None:
                        feedback_q.put("点击已处理")
                    else:
                        feedback_q.put(msg)
                elif command == "undo":
                    app.controller.undo_click()
                    feedback_q.put("已撤回一步")
                elif command == "reset":
                    app.reset()
                    feedback_q.put("已重置所有点击")
                elif command == "finish":
                    app.controller.finish_object()
                    feedback_q.put("已保存此物品结果")

    # 启动消息队列处理线程
    if q:
        threading.Thread(target=process_queue, daemon=True).start()

    # 启动主循环（这将阻塞当前线程）
    root.deiconify()
    feedback_q.put("程序已启动")
    app.mainloop()
    return app  # 返回 app 对象，以便外部调用


def run_gui_in_thread(args, cfg):
    q = queue.Queue()
    feedback_q = queue.Queue()  # 用于返回反馈信息的队列
    # 启动 GUI 线程，并返回 app 对象和控制队列
    app_thread = threading.Thread(target=start_gui, args=(args, cfg, q, feedback_q), daemon=True)
    app_thread.start()
    return q, feedback_q  # 返回用于控制的队列
