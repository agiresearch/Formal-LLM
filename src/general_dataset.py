from torchvision import transforms
from PIL import Image
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import copy


class GeneralDataset(Dataset):
    def __init__(self, folder_name, folder_path=None):
        self.folder_name = folder_name

        dir_path = folder_path + '{}/'.format(str(self.folder_name))

        # used to load the images
        def image_loader(path):
            images = []
            file_names = os.listdir(path)
            file_len = len(file_names)
            for file in range(file_len):
                temp = Image.open(path + str(file) + ".jpg").convert('RGB')
                keep = temp.copy()
                images.append(keep)
                temp.close()

            return images

        # used to load the txt files
        def text_loader(path):
            text = []
            with open(path) as f:
                lines = f.readlines()
                for line in lines:
                    text.append(line)
            f.close()
            return text

        # used to select image_loader or text loader
        def file_loader(path):
            files = []
            file_type = []
            file_names = os.listdir(path)
            for file in file_names:
                if file[-3:] == 'txt':
                    files.append(text_loader(path + file))
                    file_type.append('text')
                else:
                    files.append(image_loader(path + file + '/'))
                    file_type.append('image')

            return files, file_type

        input_path = dir_path + 'inputs/'
        output_path = dir_path + 'outputs/'

        test_path = os.listdir(output_path)

        self.input_files, self.input_file_type = file_loader(input_path)
        self.output_files, self.output_file_type = file_loader(output_path)

    def __len__(self):
        return len(self.input_files[0])

    def __getitem__(self, idx):

        # transform the images to embeddings
        transform_img = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(256),
            transforms.PILToTensor(),
        ])

        input_files = []
        output_files = []

        # add the files(txt or images) to the input and output
        for contents in self.input_files:
            content = contents[idx]
            if isinstance(content, str):
                input_files.append(content)
            else:
                input_files.append(transform_img(content))

        for contents in self.output_files:
            content = contents[idx]
            if isinstance(content, str):
                output_files.append(content)
            else:
                output_files.append(transform_img(content))

        return {'input': input_files, 'output': output_files}
