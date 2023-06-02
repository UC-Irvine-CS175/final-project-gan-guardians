'''
The purpose of augmentations is to increase the size of the training set
by applying random (or selected) transformations to the training images.

Create augmentation classes for use with the PyTorch Compose class 
that takes a list of transformations and applies them in order, which 
can be chained together simply by defining a __call__ method for each class. 
'''
import cv2
import numpy as np
import torch
from typing import Any, Tuple
import torchvision
from random import randint

class NormalizeBPS(object):
    def __call__(self, img_array) -> np.array(np.float32):
        """
        Normalize the array values between 0 - 1
        """
        # Normalizing uint numpy arrays representing images to a floating point
        # range between 0 and 1 brings the pixel values of the image to a common
        # scale that is compatible with most deep learning models.
        # Additionally, normalizing the pixel values can help to reduce the effects
        # of differences in illumination and contrast across different images, which
        # can be beneficial for model training. To normalize, we divide each pixel
        # value by the maximum value of the uint16 data type.

        # Normalize array values between 0 - 1
        img_array = img_array / np.iinfo(np.uint16).max

        # Conversion of uint16 -> float32
        img_normalized = img_array.astype(np.float32)

        # img_normalized = img_float / np.max(img_float)  # 65535.0

        return img_normalized


class ResizeBPS(object):
    def __init__(self, resize_height: int, resize_width:int):
        self._resize_height = resize_height
        self._resize_width = resize_width
    
    def __call__(self, img:np.ndarray) -> np.ndarray:
        """
        Resize the image to the specified width and height

        args:
            img (np.ndarray): image to be resized.
        returns:
            torch.Tensor: resized image.
        """
        img.resize((self._resize_height, self._resize_width), refcheck=False)
        return img


class VFlipBPS(object):
    def __call__(self, image) -> np.ndarray:
        """
        Flip the image vertically
        """
        np.flip(image, axis=0)
        return image


class HFlipBPS(object):
    def __call__(self, image) -> np.ndarray:
        """
        Flip the image horizontally
        """
        np.flip(image, axis=1)
        return image


class RotateBPS(object):
    def __init__(self, rotate: int) -> None:
        self._rotate = rotate

    def __call__(self, image) -> Any:
        '''
        Initialize an object of the Augmentation class
        Parameters:
            rotate (int):
                Optional parameter to specify a 90, 180, or 270 degrees of rotation.
        Returns:
            np.ndarray
        '''
        num_rotates = self._rotate//90

        for i in range(num_rotates):
            np.rot90(image)
        
        return image


class RandomCropBPS(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
        is made.
    """

    def __init__(self, output_height: int, output_width: int):
        self._output_height = output_height
        self._output_width = output_width

    def __call__(self, image):
        topleft_x = randint(0, len(image[0])-(self._output_width))
        topleft_y = randint(0, len(image)-(self._output_height))

        image = image[topleft_y:topleft_y+self._output_height, topleft_x:topleft_x+self._output_width]

        return image


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        if len(image.shape) == 2:
            image = image.reshape(image.shape[0], image.shape[1], 1)

        img = image.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img)
        #img_tensor = torch.from_numpy(image).unsqueeze(0)
        # image = image.transpose((2, 0, 1))
        return img_tensor


class ZoomBPS(object):
    def __init__(self, zoom: float=1) -> None:
        self.zoom = zoom

    def __call__(self, image) -> np.ndarray:
        s = image.shape
        s1 = (int(self.zoom*s[0]), int(self.zoom*s[1]))
        img = np.zeros((s[0], s[1]))
        img_resize = cv2.resize(image, (s1[1],s1[0]), interpolation = cv2.INTER_AREA)
        # Resize the image using zoom as scaling factor with area interpolation
        if self.zoom < 1:
            y1 = s[0]//2 - s1[0]//2
            y2 = s[0]//2 + s1[0] - s1[0]//2
            x1 = s[1]//2 - s1[1]//2
            x2 = s[1]//2 + s1[1] - s1[1]//2
            img[y1:y2, x1:x2] = img_resize
            return img
        else:
            return img_resize


class RescaleBPS(object):
    def __call__(self, img_array) -> np.array(np.float32):
        """
        Rescale the array values between -1 and 1
        """
        img_array = img_array / np.iinfo(np.uint16).max
        img_float = img_array.astype(np.float32)
        img_rescaled = img_float * 2 - 1
        return img_rescaled

def main():
    """Driver function for testing the augmentations. Make sure the file paths work for you."""
    # load image using cv2
    img_key = 'P280_73668439105-F5_015_023_proj.tif'
    img_array = cv2.imread(img_key, cv2.IMREAD_ANYDEPTH)
    print(img_array.shape, img_array.dtype)
    test_resize = ResizeBPS(500, 500)
    type(test_resize)

if __name__ == "__main__":
    main()