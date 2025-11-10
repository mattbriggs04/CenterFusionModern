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
        This method correctly replicates the logic from test.py's PrefetchDataset.
        """
        print(f"--- Processing Sample {img_idx} ---")
        
        # 1. Get image info, path, and raw image
        img_id = self.dataset.images[img_idx]
        img_info = self.dataset.coco.loadImgs(ids=[img_id])[0]
        img_path = os.path.join(self.dataset.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error: Could not load image at {img_path}")
            return

        # 2. Get calibration
        input_meta = {}
        if 'calib' in img_info:
            input_meta['calib'] = img_info['calib']
        
        # 3. Call pre_process to generate 'meta' (with 'c', 's', etc.)
        # This is the step that was missing.
        scale = self.opt.test_scales[0] # We only use one scale
        images_tensor, meta = self.detector.pre_process(image, scale, input_meta)

        # 4. Load Point Cloud
        # This function *requires* the 'meta' dict from pre_process
        pc_2d, pc_N, pc_dep, pc_3d = self.dataset._load_pc_data(
            image, img_info, meta['trans_input'], meta['trans_output']
        )
        
        # 5. Convert data to Tensors on the correct device
        img_tensor = images_tensor.to(self.opt.device)
        pc_dep_tensor = torch.from_numpy(pc_dep).unsqueeze(0).to(self.opt.device)
        
        # 6. Run Inference (process)
        print("Running model inference...")
        with torch.no_grad():
            output, dets = self.detector.process(
                img_tensor, pc_dep=pc_dep_tensor, meta=meta
            )
        
        # 7. Run Post-processing
        print("Post-processing detections...")
        results = self.detector.post_process(dets, meta)

        # 8. Run Drawing
        print("Drawing bounding boxes...")
        self.detector.show_results(self.debugger, image, results)
        
        # 9. Get the final image from the debugger's canvas
        if 'ddd_pred' in self.debugger.imgs:
            drawn_img = self.debugger.imgs['ddd_pred']
        else:
            drawn_img = self.debugger.imgs['generic']
        
        print("Inference complete.")
        return drawn_img

    def run_headless(self, img_idx: int):
        """
        Runs the full inference and drawing pipeline on a single image.
        This method correctly replicates the logic from test.py's PrefetchDataset.
        """
        print(f"--- Processing Sample {img_idx} ---")
        
        # 1. Get image info, path, and raw image
        img_id = self.dataset.images[img_idx]
        img_info = self.dataset.coco.loadImgs(ids=[img_id])[0]
        img_path = os.path.join(self.dataset.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error: Could not load image at {img_path}")
            return

        # 2. Get calibration
        input_meta = {}
        if 'calib' in img_info:
            input_meta['calib'] = img_info['calib']
        
        # 3. Call pre_process to generate 'meta' (with 'c', 's', etc.)
        scale = self.opt.test_scales[0] # We only use one scale
        images_tensor, meta = self.detector.pre_process(image, scale, input_meta)

        # 4. Load Point Cloud
        pc_2d, pc_N, pc_dep, pc_3d = self.dataset._load_pc_data(
            image, img_info, meta['trans_input'], meta['trans_output']
        )
        
        # 5. Convert data to Tensors on the correct device
        img_tensor = images_tensor.to(self.opt.device)
        pc_dep_tensor = torch.from_numpy(pc_dep).unsqueeze(0).to(self.opt.device)
        
        # 6. Run Inference (process)
        print("Running model inference...")
        with torch.no_grad():
            output, dets = self.detector.process(
                img_tensor, pc_dep=pc_dep_tensor, meta=meta
            )
        
        # 7. Run Post-processing
        print("Post-processing detections...")
        results = self.detector.post_process(dets, meta)

        # --- FIX: Replaced self.detector.show_results ---
        # 8. Run Drawing (copied from detector.show_results)
        print("Drawing bounding boxes...")
        self.debugger.add_img(image, img_id='generic')

        # Draw 3D Detections
        if len(results) > 0 and 'dep' in results[0] and 'alpha' in results[0] and 'dim' in results[0]:
            self.debugger.add_3d_detection(
                image, False, results, meta['calib'],
                vis_thresh=self.opt.vis_thresh, img_id='ddd_pred')
            
            self.debugger.add_bird_view(
                results, vis_thresh=self.opt.vis_thresh,
                img_id='bird_pred', cnt=self.detector.cnt)
        
        # 9. Get the final image from the debugger's canvas
        if 'ddd_pred' in self.debugger.imgs:
            drawn_img = self.debugger.imgs['ddd_pred']
        else:
            # Fallback if 3D drawing failed
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
    result_img = vis.run_headless(opt.sample_id)
    
    # determine save path
    save_path = opt.save_path
    if save_path == '':
        save_path = f'./sample_{opt.sample_id:04d}_vis.jpg'

    # save the final image
    cv2.imwrite(save_path, result_img)
    
    # get absolute path for a clear message
    final_path = os.path.abspath(save_path)
    print(f"\nSuccessfully saved visualization to:\n{final_path}")