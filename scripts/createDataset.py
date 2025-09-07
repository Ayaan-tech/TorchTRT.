import os, random, cv2, torch, numpy as np, matplotlib.pyplot as plt

class Dataset:
    def __init__(self , root_dir=None  ,  max_images = 150, target_size=(640, 640)):
        self.root_dir = root_dir
        self.max_images = max_images
        self.target_size = target_size

    def find_image_paths(self):
        paths= []
        for root, dirs, files in os.walk(self.root_dir):
            for filename in files:
                if filename.endswith((".png", ".jpg", ".jpeg")):
                    paths.append(os.path.join(root, filename))
        return paths
    def prepare_images(self):
        image_paths = self.find_image_paths()
        image_paths = list(image_paths[:self.max_images])
        images = []
        tensors = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img_rgb, self.target_size)
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                tensors.append(img_tensor)
                images.append(img)
        return images, tensors
    def show_images(self , images, n=None , cols=4, figsize=(12, 8)):
        n = min(n, len(images))
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if isinstance(axes, (list, tuple,)) or hasattr(axes, "flatten") else [axes]
        if rows* cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        for i in range(n):
            axes[i].imshow(images[i])
            axes[i].axis('off')
        for j in range(n , len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        plt.show()