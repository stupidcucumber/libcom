import torch
import torchvision
from libcom.utils.model_download import download_pretrained_model
from libcom.utils.process_image import *
from libcom.utils.environment import *
from libcom.opa_score.source import ObjectPlaceNet
import torch 
import os
import torchvision.transforms as transforms

cur_dir   = os.path.dirname(os.path.abspath(__file__))
model_set = ['SimOPA'] 


class OPAScoreModel:
    """
    Foreground object search score prediction model.

    Args:
        device (str | torch.device): gpu id
        model_type (str): predefined model type.
        kwargs (dict): other parameters for building model
    
    Examples:
        >>> from libcom import OPAScoreModel
        >>> from libcom.utils.process_image import make_image_grid
        >>> import cv2
        >>> net = OPAScoreModel(device=0, model_type='SimOPA')
        >>> comp_img  = '../tests/source/composite/1bb218825e370c51_m02p5f1q_4a9db762_30.png'
        >>> comp_mask = '../tests/source/composite_mask/1bb218825e370c51_m02p5f1q_4a9db762_30.png'
        >>> score = net(comp_img, comp_mask)
        >>> grid_img  = make_image_grid([comp_img, comp_mask], text_list=[f'opa_score:{score:.2f}', 'composite-mask'])
        >>> cv2.imwrite('../docs/_static/image/opascore_result1.jpg', grid_img)

    Expected result:

    .. image:: _static/image/opascore_result1.jpg
        :scale: 50 %

            
    """
    def __init__(self, device=0, model_type='SimOPA', **kwargs):
        assert model_type in model_set, f'Not implementation for {model_type}'
        self.model_type = model_type
        self.option = kwargs
        weight_path = os.path.join(cur_dir, 'pretrained_models', 'SimOPA.pth')
        download_pretrained_model(weight_path)
        self.device = check_gpu_device(device)
        self.build_pretrained_model(weight_path)
        self.build_data_transformer()

    def build_pretrained_model(self, weight_path):
        model = ObjectPlaceNet(False)
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))
        self.model = model.to(self.device).eval()
        
    def build_data_transformer(self):
        self.image_size = 256
        self.transformer = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
    
    def inputs_preprocess(self, composite_image, composite_mask):
        img  = read_image_pil(composite_image)
        img  = self.transformer(img)
        mask = read_mask_pil(composite_mask)
        mask = self.transformer(mask)
        cat_img = torch.cat([img, mask], dim=0)
        cat_img = cat_img.unsqueeze(0).to(self.device)
        return cat_img
    
    def outputs_postprocess(self, outputs):
        score   = torch.softmax(outputs, dim=-1)[0, 1].cpu().item()
        return score
    
    @torch.no_grad()
    def __call__(self, composite_image, composite_mask):
        """
        Predicting the object placement assessment (opa) score for the given composite image, which evaluates the rationality of foreground object placement. Called in __call__ function.

        Args:
            composite_image (str | numpy.ndarray): The path to composite image or the compposite image in ndarray form.
            composite_mask (str | numpy.ndarray): Mask of composite image which indicates the foreground object region in the composite image.
        
        Returns:
            opa_score (float): Predicted opa score ranges from 0 to 1, where a larger score indicates more reasonable placement.

        """
        inputs    = self.inputs_preprocess(composite_image, composite_mask)
        outputs   = self.model(inputs)
        preds     = self.outputs_postprocess(outputs)
        return preds