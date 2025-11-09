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
    single nuScenes sample.
    
    This class re-uses the Detector's internal methods for
    processing and drawing.
    """

    def __init__(self, opt):
        print("Initializing visualizer...")
        self.opt = opt
        self.opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Load the dataset
        print(f"Loading dataset split: {self.opt.val_split}")
        Dataset = dataset_factory[self.opt.dataset]
        self.dataset = Dataset(self.opt, self.opt.val_split)
        
        # Initialize the Detector. This will:
        # 1. Create the model
        # 2. Load the weights from opt.load_model
        # 3. Create its own internal Debugger
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
        # 'sample' is a dict with tensors: 'image', 'pc_dep', 'calib', 'meta'
        sample = self.dataset[img_idx]

        # 2. Get the original image for drawing
        img_path = self.dataset.get_image_path(img_idx)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not load image at {img_path}")
            return

        # 3. Add a batch dimension and send to device
        img_tensor = sample['image'].unsqueeze(0).to(self.opt.device)
        pc_dep_tensor = sample['pc_dep'].unsqueeze(0).to(self.opt.device)
        
        # 'meta' is used by process and post_process
        meta = sample['meta'] 

        # 4. Run Inference (process)
        print("Running model inference...")
        # We pass 'meta' so the model can get calibration data
        with torch.no_grad():
            output, dets, forward_time = self.detector.process(
                img_tensor, pc_dep=pc_dep_tensor, meta=meta
            ) #
        
        # 5. Run Post-processing
        # This converts detections to the original image scale
        print("Post-processing detections...")
        results = self.detector.post_process(dets, meta) #

        # 6. Run Drawing
        # This uses the detector's *internal* debugger to draw
        print("Drawing bounding boxes...")
        self.detector.show_results(self.debugger, img, results) #
        
        # 7. Get the final image from the debugger's canvas
        # show_results draws on 'ddd_pred' or 'generic'
        if 'ddd_pred' in self.debugger.imgs:
            drawn_img = self.debugger.imgs['ddd_pred']
        else:
            drawn_img = self.debugger.imgs['generic']
        
        print("Inference complete.")
        return drawn_img

    def get_dataset_size(self):
        return len(self.dataset)

if __name__ == "__main__":
    
    # Use argparse to add our new, custom arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_id', type=int, default=10,
                        help='Index of the sample to visualize')
    parser.add_argument('--save_path', type=str, default='',
                        help='Path to save the output image (default: sample_XX.jpg)')
    
    # Use the repo's opts.py to parse all known arguments
    # AND our custom ones.
    # The 'parse' method takes an 'args' list, not a parser object
    # We must manually separate our args from the repo's args.
    
    # This is a bit of a hack to get args from both
    known_args, unknown_args = parser.parse_known_args()
    
    # Pass the remaining (unknown) args to the main opts parser
    opt = opts().parse(unknown_args)
    
    # Add our custom args to the opt namespace
    opt.sample_id = known_args.sample_id
    opt.save_path = known_args.save_path
    
    # Create the visualizer
    vis = NuScenesVisualizer(opt)
    
    # Run visualization
    result_img = vis.run(opt.sample_id)
    
    # Determine save path
    save_path = opt.save_path
    if save_path == '':
        save_path = f'./sample_{opt.sample_id:04d}_vis.jpg'

    # Save the final image
    cv2.imwrite(save_path, result_img)
    
    # Get absolute path for a clear message
    final_path = os.path.abspath(save_path)
    print(f"\nSuccessfully saved visualization to:\n{final_path}")