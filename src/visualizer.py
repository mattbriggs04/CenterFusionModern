import torch
import cv2
import os
import argparse

from lib.opts import opts
from lib.detector import Detector
from lib.dataset.dataset_factory import dataset_factory
from lib.dataset.datasets.nuscenes import nuScenes 

class NuScenesVisualizer():
    """
    A class to load a model and visualize its output on a
    single nuScenes sample by drawing bounding boxes.
    """

    def __init__(self, opt):
        print("Initializing visualizer...")
        self.opt = opt
        self.opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # load the dataset
        Dataset = dataset_factory[self.opt.dataset]
        self.opt = opts().update_dataset_info_and_set_heads(self.opt, Dataset)
        print(f"Loading dataset split: {self.opt.val_split}")
        self.dataset = Dataset(self.opt, self.opt.val_split)
        
        # Initialize the Detector to create model, load weights, and create debugger
        print("Loading model and detector...")
        self.detector = Detector(self.opt)
        
        # Get the debugger from the detector
        self.debugger = self.detector.debugger


    def run(self, img_idx: int):
        """
        Runs the full inference and drawing pipeline on a single image
        by correctly using the detector.run() method.
        """
        print(f"--- Processing Sample {img_idx} ---")
        
        # 1. Get image info and path, just like test.py
        img_id = self.dataset.images[img_idx]
        img_info = self.dataset.coco.loadImgs(ids=[img_id])[0]
        img_path = os.path.join(self.dataset.img_dir, img_info['file_name'])

        # 2. Get metadata (calib)
        input_meta = {}
        if 'calib' in img_info:
            input_meta['calib'] = img_info['calib']
        
        # 3. Call detector.run()
        # This single function will do all the steps you were
        # trying to do manually (pre-process, process, post-process,
        # and draw).
        print("Running full detection pipeline...")
        # We pass the path and meta, not tensors.
        ret = self.detector.run(img_path, input_meta)    
        
        # 4. Get the final image from the detector's internal debugger
        # The 'show_results' method (called by run) draws
        # on 'ddd_pred' or 'generic'.
        if 'ddd_pred' in self.detector.debugger.imgs:
            drawn_img = self.detector.debugger.imgs['ddd_pred']
        else:
            drawn_img = self.detector.debugger.imgs['generic']
        
        print("Inference complete.")
        return drawn_img

if __name__ == "__main__":
    
    # use argparse to add our new, custom arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_id', type=int, default=10,
                        help='Index of the sample to visualize')
    parser.add_argument('--save_path', type=str, default='',
                        help='Path to save the output image (default: sample_XX.jpg)')
    
    # this is a bit of a hack to get args from both
    known_args, unknown_args = parser.parse_known_args()
    
    # pass the remaining (unknown) args to the main opts parser
    opt = opts().parse(unknown_args)
    
    # add our custom args to the opt namespace
    opt.sample_id = known_args.sample_id
    opt.save_path = known_args.save_path
    
    # create the visualizer
    vis = NuScenesVisualizer(opt)
    
    # run visualization
    result_img = vis.run(opt.sample_id)
    
    # determine save path
    save_path = opt.save_path
    if save_path == '':
        save_path = f'./sample_{opt.sample_id:04d}_vis.jpg'

    # save the final image
    cv2.imwrite(save_path, result_img)
    
    # get absolute path for a clear message
    final_path = os.path.abspath(save_path)
    print(f"\nSuccessfully saved visualization to:\n{final_path}")