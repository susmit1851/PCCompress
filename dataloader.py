from utils import *

def triangle_sampling(v1, v2, v3, n_points):
    v1 = np.array(v1)
    v2 = np.array(v2)
    v3 = np.array(v3)

    u1 = np.random.rand(n_points,1)
    u2 = np.random.rand(n_points, 1)
    mask = u1 + u2 > 1
    u1[mask] = 1 - u1[mask]
    u2[mask] = 1 - u2[mask]
    a = v2 - v1
    b = v3 - v1
    return a * u1 + b * u2 + v1


def parse_off(filepath, num_pts_triangle=300):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    values = lines[1].strip().split()
    num_vertices = int(values[0])
    num_faces = int(values[1])

    vertices = []
    for i in range(num_vertices):
        v = list(map(float, lines[2+i].strip().split()))
        vertices.append(v)
    for j in range(num_faces):
        face = lines[2+num_vertices+j].strip().split()
        f = list(map(int, face))
        f = f[1:]
        v1 = vertices[f[0]]
        v2 = vertices[f[1]]
        v3 = vertices[f[2]]
        sampled_pts = triangle_sampling(v1,v2,v3, num_pts_triangle)
        vertices.extend(sampled_pts)


    return vertices

## For OOD Robustness, we will skip 'monitor' and 'bed' classes in train data
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







class PCDataset(Dataset):
    def __init__(self, data_dir, split, num_pts_triangle=500):
        self.file_paths = []
        self.num_pts_triangle = num_pts_triangle
        self.split = split
        skip_classes = {'monitor', 'bed'} if split == 'train' else set()

        for class_name in os.listdir(data_dir):
            if class_name in skip_classes:
                continue
            class_path = os.path.join(data_dir, class_name, split)
            for fname in os.listdir(class_path):
                if fname.endswith('.off'):
                    self.file_paths.append(os.path.join(class_path, fname))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        point_cloud = parse_off(file_path, num_pts_triangle=self.num_pts_triangle)
        point_cloud = np.array(point_cloud, dtype=np.float32)
        return torch.from_numpy(point_cloud)

def getDataloader(data_dir, split, batch_size = 8, num_pts_triangle = 500, shuffle = True):
    dataset = PCDataset(data_dir=data_dir, split=split, num_pts_triangle=num_pts_triangle)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)