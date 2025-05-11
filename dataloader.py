# from utils import *

# def triangle_sampling(v1, v2, v3, n_points):
#     v1 = np.array(v1)
#     v2 = np.array(v2)
#     v3 = np.array(v3)

#     u1 = np.random.rand(n_points,1)
#     u2 = np.random.rand(n_points, 1)
#     mask = u1 + u2 > 1
#     u1[mask] = 1 - u1[mask]
#     u2[mask] = 1 - u2[mask]
#     a = v2 - v1
#     b = v3 - v1
#     return a * u1 + b * u2 + v1


# def parse_off(filepath, num_pts_triangle=300):
#     with open(filepath, 'r') as f:
#         lines = f.readlines()

#     values = lines[1].strip().split()
#     num_vertices = int(values[0])
#     num_faces = int(values[1])

#     vertices = []
#     for i in range(num_vertices):
#         v = list(map(float, lines[2+i].strip().split()))
#         vertices.append(v)
#     for j in range(num_faces):
#         face = lines[2+num_vertices+j].strip().split()
#         f = list(map(int, face))
#         f = f[1:]
#         v1 = vertices[f[0]]
#         v2 = vertices[f[1]]
#         v3 = vertices[f[2]]
#         sampled_pts = triangle_sampling(v1,v2,v3, num_pts_triangle)
#         vertices.extend(sampled_pts)


#     return vertices

# ## For OOD Robustness, we will skip 'monitor' and 'bed' classes in train data
# class PCDataset(Dataset):
#     def __init__(self, data_dir, split, num_pts_triangle=500):
#         self.data_dir = data_dir
#         self.num_pts_triangle = num_pts_triangle
#         self.split = split
#         self.train_data = []
#         self.test_data = []
#         for classes in os.listdir(data_dir):
#             if classes == 'monitor' or 'bed':
#                 class_path = os.path.join(data_dir, classes)
#                 off_dir = os.path.join(class_path, self.split)
#                 for files in os.listdir(off_dir):
#                     off_file_path = os.path.join(off_dir, files)
#                     self.test_data.append(parse_off(off_file_path, num_pts_triangle=self.num_pts_triangle))

#             else:
#                 class_path = os.path.join(data_dir, classes)
#                 off_dir = os.path.join(class_path, self.split)
#                 for files in os.listdir(off_dir):
#                     off_file_path = os.path.join(off_dir, files)
#                     if self.split == 'train':
#                         self.train_data.append(parse_off(off_file_path, num_pts_triangle=self.num_pts_triangle))
#                     else:
#                         self.test_data.append(parse_off(off_file_path, num_pts_triangle=self.num_pts_triangle))
            



#     def __len__(self):
#         if self.split == 'train':
#             return len(self.train_data)
#         else:
#             return len(self.test_data)
        
    
#     def __getitem__(self, index):
#         if self.split == 'train':
#             return self.train_data[index]
#         else:
#             return self.test_data[index]

# def getDataloader(data_dir, split, batch_size = 8, num_pts_triangle = 500, shuffle = True):
#     dataset = PCDataset(data_dir=data_dir, split=split, num_pts_triangle=num_pts_triangle)
#     return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)




from utils import *
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import concurrent.futures

def triangle_sampling(v1, v2, v3, n_points):
    # Vectorized computation
    v1, v2, v3 = map(np.array, [v1, v2, v3])
    u1 = np.random.rand(n_points, 1)
    u2 = np.random.rand(n_points, 1)
    mask = u1 + u2 > 1
    u1[mask], u2[mask] = 1 - u1[mask], 1 - u2[mask]
    return (v2 - v1) * u1 + (v3 - v1) * u2 + v1

def parse_off(filepath, num_pts_triangle=300):
    # Use numpy for faster file reading and processing
    with open(filepath, 'r') as f:
        # Skip the first line (OFF)
        next(f)
        num_vertices, num_faces, _ = map(int, next(f).split())
        
        # Read vertices directly into numpy array
        vertices = np.array([
            list(map(float, next(f).split())) 
            for _ in range(num_vertices)
        ])
        
        # Process faces and sample points
        sampled_points = []
        for _ in range(num_faces):
            _, v1_idx, v2_idx, v3_idx = map(int, next(f).split())
            v1, v2, v3 = vertices[v1_idx], vertices[v2_idx], vertices[v3_idx]
            sampled_points.append(triangle_sampling(v1, v2, v3, num_pts_triangle))
        
        return np.vstack([vertices] + sampled_points)

class PCDataset(Dataset):
    def __init__(self, data_dir, split, num_pts_triangle=500):
        self.data_dir = Path(data_dir)
        self.num_pts_triangle = num_pts_triangle
        self.split = split
        self.data = []
        
        # Define excluded classes
        excluded_classes = {'monitor', 'bed'}
        
        # Parallel processing of files
        def process_file(file_path):
            return parse_off(str(file_path), num_pts_triangle=self.num_pts_triangle)
        
        # Collect all valid file paths
        file_paths = []
        for class_path in self.data_dir.iterdir():
            if split == 'train' and class_path.name in excluded_classes:
                continue
                
            off_dir = class_path / split
            if off_dir.exists():
                file_paths.extend(off_dir.glob('*.off'))
        
        # Process files in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            self.data = list(executor.map(process_file, file_paths))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

def getDataloader(data_dir, split, batch_size=8, num_pts_triangle=500, shuffle=True, num_workers=4):
    dataset = PCDataset(data_dir=data_dir, split=split, num_pts_triangle=num_pts_triangle)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True
    )