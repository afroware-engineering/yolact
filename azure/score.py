import os
import sys
import logging
import json
import base64
import numpy as np
import cv2
import torch
import types



net = None
model_path = None

def init():
    global net, yolact_eval, cfg, model_path

    logging.info("YOLACT init started")

    model_base = os.getenv("AZUREML_MODEL_DIR")
    model_folder = "model_folder_name" 
    model_path = os.path.join(model_base, model_folder)
    weights_file_name = ""
    config_name = ""
    
    if model_path not in sys.path:
        sys.path.insert(0, model_path)

    # Import here after sys.path is set
    from yolact640.yolact import Yolact
    from yolact640.data import set_cfg, cfg as local_cfg
    import yolact640.eval as local_eval

    yolact_eval = local_eval
    cfg = local_cfg

    set_cfg(config_name)
    cfg.mask_proto_debug = False

    net = Yolact()
    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = os.path.join(model_path, "weights", weights_file_name)
    net.load_state_dict(torch.load(weights_path, map_location=map_location))
    net.eval()
    net.detect.use_fast_nms = True

    if torch.cuda.is_available():
        net = net.cuda()

    logging.info("YOLACT model loaded successfully")


def run(raw_data):
    global net
    logging.info("YOLACT run: request received")

    try:
        request = json.loads(raw_data)
        
        # Expecting base64-encoded image in JSON under 'image'
        image_b64 = request["image"]
        image_bytes = base64.b64decode(image_b64)

        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Set args
        yolact_eval.args = types.SimpleNamespace(
            display_lincomb=False,
            mask_proto_debug=False,
            display_masks=True,
            display_bboxes=True,
            display_text=True,
            display_scores=True,
            crop=False,
            score_threshold=0.15,
            top_k=15,
        )

        # Run inference
        output_img = yolact_eval.evalimage_from_array(net, img)

        # Convert back to BGR for OpenCV and encode as JPEG
        output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", output_bgr)
        output_base64 = base64.b64encode(buffer).decode("utf-8")

        return { "image": output_base64 }

    except Exception as e:
        logging.error(f"YOLACT run failed: {str(e)}")
        return { "error": str(e) }
