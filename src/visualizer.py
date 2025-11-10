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
        Runs the full inference and drawing pipeline on a single image.
        """
        print(f"--- Processing Sample {img_idx} ---")
        
        # 1. Get the data sample from the dataset loader
        # 'sample' is a dict with ALL data as numpy arrays
        sample = self.dataset[img_idx]

        # 2. Get the original image for drawing
        img_id = self.dataset.images[img_idx]
        img_info = self.dataset.coco.loadImgs(ids=[img_id])[0]
        img_path = os.path.join(self.dataset.img_dir, img_info['file_name'])
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not load image at {img_path}")
            return

        # 3. Add a batch dimension and send to device
        # --- THIS IS THE FIX ---
        # Convert NumPy arrays to PyTorch tensors before calling .unsqueeze()
        img_tensor = torch.from_numpy(sample['image']).unsqueeze(0).to(self.opt.device)
        pc_dep_tensor = torch.from_numpy(sample['pc_dep']).unsqueeze(0).to(self.opt.device)
        
        # 4. Create the 'meta' dict needed by the detector
        # All this info is in the 'sample' dict at the top level.
        meta = {
            'calib': sample['calib'],
            'c': sample['c'],
            's': sample['s'],
            'out_height': self.opt.output_h,
            'out_width': self.opt.output_w,
            'height': img.shape[0],
            'width': img.shape[1]
        }

        # 5. Run Inference (process)
        print("Running model inference...")
        with torch.no_grad():
            output, dets, forward_time = self.detector.process(
                img_tensor, pc_dep=pc_dep_tensor, meta=meta
            )
        
        # 6. Run Post-processing
        print("Post-processing detections...")
        results = self.detector.post_process(dets, meta)

        # 7. Run Drawing
        print("Drawing bounding boxes...")
        self.detector.show_results(self.debugger, img, results)
        
        # 8. Get the final image from the debugger's canvas
        if 'ddd_pred' in self.debugger.imgs:
            drawn_img = self.debugger.imgs['ddd_pred']
        else:
            drawn_img = self.debugger.imgs['generic']
        
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