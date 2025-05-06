from predictor import *

from scribbleprompt.models.network import UNet
from scribbleprompt.models.unet import rescale_inputs, prepare_inputs

device = 'cuda' if torch.cuda.is_available() else 'cpu'

RES = 512

file_dir = pathlib.Path(os.path.dirname(__file__))
exp_dir = file_dir / "../../checkpoints"

file_dir = pathlib.Path(os.path.dirname(__file__))
default_model = 'dune_seg_model'

model_dict = {
    'ScribblePrompt-Unet': 'ScribblePrompt_unet_v1_nf192_res128.pt',
    'dune_seg_model': 'max-val_od-dice_score.pt'
}


class Predictor:
    """
    Wrapper for ScribblePrompt-UNet model
    """

    def __init__(self, path: str, verbose: bool = True):

        assert path.exists(), f"Checkpoint {path} does not exist"

        self.path = path
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_model()
        self.load()
        self.model.eval()
        self.to_device()

    def build_model(self):
        """
        Build the model
        """
        self.model = UNet(
            in_channels=5,
            out_channels=1,
            features=[192, 192, 192, 192],
        )

    def load(self):
        """
        Load the state of the model from a checkpoint file.
        """
        with (self.path).open("rb") as f:
            state = torch.load(f, map_location=self.device)
            self.model.load_state_dict(state["model"], strict=True)
            if self.verbose:
                print(
                    f"Loaded checkpoint from {self.path} to {self.device}"
                )

    def to_device(self):
        """
        Move the model to cpu or gpu
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def predict(self, prompts: Dict[str, any], img_features: Optional[torch.Tensor] = None,
                multimask_mode: bool = False):
        """
        Make predictions!

        Returns:
            mask (torch.Tensor): H x W
            img_features (torch.Tensor): B x 1 x H x W (for SAM models)
            low_res_mask (torch.Tensor): B x 1 x H x W logits
        """
        if self.verbose:
            print("point_coords", prompts.get("point_coords", None))
            print("point_labels", prompts.get("point_labels", None))
            print("box", prompts.get("box", None))
            print("img", prompts.get("img").shape, prompts.get("img").min(), prompts.get("img").max())
            if prompts.get("scribbles") is not None:
                print("scribbles", prompts.get("scribbles", None).shape, prompts.get("scribbles").min(),
                      prompts.get("scribbles").max())

        original_shape = prompts.get('img').shape[-2:]

        # Rescale to 128 x 128
        prompts = rescale_inputs(prompts)

        # Prepare inputs for ScribblePrompt unet (1 x 5 x 128 x 128)
        x = prepare_inputs(prompts).float()

        with torch.no_grad():
            yhat = self.model(x.to(self.device)).cpu()

        mask = torch.sigmoid(yhat)

        # Resize for app resolution
        mask = F.interpolate(mask, size=original_shape, mode='bilinear').squeeze()

        # mask: H x W, yhat: 1 x 1 x H x W
        return mask, None, yhat


def load_model(exp_key: str = default_model):
    """
    exp_key: ScribblePrompt
    model_dict = {
    # 'ScribblePrompt-Unet': 'ScribblePrompt_unet_v1_nf192_res128.pt'
    'ScribblePrompt-Unet': 'max-val_od-dice_score.pt'
    # 'ScribblePrompt-Unet': 'epoch-400.pt'
    }
    """
    fpath = exp_dir / model_dict.get(exp_key)
    exp = Predictor(fpath)
    return exp, None


separate_scribble_masks = np.zeros((2, RES, RES), dtype=np.float32)
last_scribble_mask = np.zeros((RES, RES), dtype=np.float32)

click_coords = []
click_labels = []
bbox_coords = []
# Load default model
predictor = load_model()[0]
img_features = None  # For SAM models
best_mask = None
low_res_mask = None

scribble_img = {
    'image': np.zeros((RES, RES, 3), dtype=np.uint32),
    'mask': np.zeros((RES, RES, 3), dtype=np.uint32)
}
click_img = {
    'image': np.zeros((RES, RES, 3), dtype=np.uint32),
    'mask': np.zeros((RES, RES, 3), dtype=np.uint32)
}
input_img = None
output_img = None
binary_checkbox = True
multimask_mode = True

current_label = 1


def init_current_label():
    global current_label
    current_label = 1
    pass


def change_current_label():
    global current_label
    if current_label == 1:
        current_label = 0
    elif current_label == 0:
        current_label = 1
    return current_label


def clear_all_history(img):
    if img is not None:
        input_shape = img.shape[:2]
    else:
        input_shape = (RES, RES)

    temp_dict = {
        'image': img,
        'mask': np.zeros((RES, RES, 3), dtype=np.uint32)
    }
    return (temp_dict, temp_dict, [], [], [], [], np.zeros((2,) + input_shape, dtype=np.float32),
            np.zeros(input_shape, dtype=np.float32), None, None, None)


def change_input_img(img):
    global input_img, click_img, scribble_img, output_img, click_coords, click_labels, \
        bbox_coords, separate_scribble_masks, last_scribble_mask, best_mask, low_res_mask, img_features

    input_img = img
    (click_img, scribble_img, output_img, click_coords, click_labels,
     bbox_coords, separate_scribble_masks, last_scribble_mask, best_mask, low_res_mask,
     img_features) = clear_all_history(input_img)
    return input_img
    pass


def click_seg(coords):
    global click_img, scribble_img, output_img, best_mask, low_res_mask, img_features, \
        click_coords, click_labels, bbox_coords, separate_scribble_masks, last_scribble_mask, current_label

    # current_label = label
    autopredict_checkbox = False

    (click_img, scribble_img, output_img, best_mask, low_res_mask, img_features, click_coords, click_labels,
     bbox_coords, separate_scribble_masks, last_scribble_mask) \
        = get_select_coords(predictor, input_img, best_mask, low_res_mask, click_coords, click_labels, bbox_coords,
                            separate_scribble_masks, last_scribble_mask, scribble_img, img_features,
                            output_img, binary_checkbox, multimask_mode, autopredict_checkbox, coords, current_label)

    return f"coords now: {click_coords}, labels now: {click_labels}"
    pass


def get_output():
    global click_img, scribble_img, output_img, best_mask, low_res_mask, img_features, \
        separate_scribble_masks, last_scribble_mask

    (click_img, scribble_img, output_img, best_mask, low_res_mask, img_features,
     separate_scribble_masks, last_scribble_mask) \
        = refresh_predictions(predictor, input_img, click_coords, click_labels, bbox_coords, current_label,
                              scribble_img, separate_scribble_masks, last_scribble_mask,
                              low_res_mask, img_features, binary_checkbox, multimask_mode)
    return output_img[0]
    pass


def paint_seg(painting_img):
    global scribble_img
    scribble_img.update({'mask': painting_img})

    return scribble_img
    pass


if __name__ == "__main__":
    pass
