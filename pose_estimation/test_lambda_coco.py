import pickle
import torch
from pose_estimation.pose_hrnet_se_lambda import get_pose_net
cfg = pickle.load(open('pose_estimation/lambda_coco.cfg', 'rb'))
model = get_pose_net(cfg, is_train=False)
model_object = torch.load('pose_estimation/weights/checkpoint_103.pth')
model.load_state_dict(model_object['latest_state_dict'], strict=False)
model = torch.nn.DataParallel(model).cuda()
model.eval()