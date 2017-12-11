"""
NLM Algorithm for Image Processing Coursework.
Code by Ben Hadfield.

Algorithm implemented from IPOL paper:
http://www.ipol.im/pub/art/2011/bcm_nlm/article.pdf

Python -> Cython conversion (for speed up) informed by a variety of sources,
including:
 - http://docs.cython.org/en/latest/
 - http://cython.readthedocs.io/en/latest/src/userguide/extension_types.html
 - https://github.com/scipy/scipy
"""

import uuid
import numpy as np
from termcolor import cprint
cimport numpy as np
from PIL import Image
from libc.math cimport exp


# The Python min and max functions works on Python objects, and so for int
# checks we can get a speed up by creating our own functions.
cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b

# Define custom image type.
ctypedef np.float32_t NLMIMGTYPE


# Validation checks (for user input).
def _throw_or_return(bint result, bint throw, error=ValueError):
    """
    Takes `result` and `throw`, and returns `True` if result is `True`, else
    raises `error` if `throw` is `True`, else returns `False`.
    """
    if throw and not result:
        raise error()
    return result


def is_gt(*args, float threshold=0, bint throw=True):
    """Checks `test` is strictly larger than `threshold`."""
    result = all([test > threshold for test in args])
    return _throw_or_return(result, throw)


def is_lt(*args, float threshold=0, bint throw=True):
    """Checks `test` is strictly less than than `threshold`."""
    result = all([test < threshold for test in args])
    return _throw_or_return(result, throw)


def is_int(*args, bint throw=True):
    """Checks all args are instances of `int`."""
    result = all([isinstance(test, int) for test in args])
    return _throw_or_return(result, throw)


# Base image class.
cdef class ImageModel:
    """
    Base class for image models.
    Sets `self.pixels` from the image path path given to the constructor, and
    implements the `run` and `setup` stub methods, intended to be overridden.
    """
    cdef public np.ndarray pixels

    def __init__(self, *args, **kwargs):
        # Get the image either as a keyword param or the first argument.
        image = kwargs.get('image') or args[0]
        if not image:
            raise AttributeError('No image supplied to ModelImage instance.')
        # Open the specified image and convert to greyscale.
        image = Image.open(image).convert('L')
        self.pixels = ImageModel.pixel_matrix(image.getdata())

    @staticmethod
    def pixel_matrix(image):
        """
        Takes an array and returns a 2D matrix of pixels.
        :param image PIL.Image instance
        """
        cdef int rows, cols
        rows = image.size[1]
        cols = image.size[0]

        matrix = np.array(image)
        matrix.shape = rows, cols
        return matrix

    @staticmethod
    def setup():
        """
        Used to initialise a new instance of the model.
        This method should take user input (in needed) and return a model
        instance.
        """
        raise NotImplementedError

    def run(self):
        """Should run the algorithm and return an Image instance."""
        raise NotImplementedError

    def save(self, image):
        """Takes an image and saves it to `img/out/`."""
        cdef str uid = str(uuid.uuid4())[:5]
        cdef str ext = 'jpg'
        cdef str name = self.__class__.__name__
        image.save('img/out/{name}-{hash}.{ext}'.format(
            name=name, hash=uid, ext=ext))
        return uid


# NLM implementation.
cdef class NLM(ImageModel):
    @staticmethod
    def setup():
        cdef NLM model
        cdef str default_image = 'alleyNoisy_sigma20.png'
        try:
            image_name = input('Enter the image name (default: {}).\n'
                               ' -> img/in/'.format(default_image))
            if not image_name:
                cprint('Using default image: {}'
                       .format(default_image), 'yellow')
            patch_radius = int(input('Enter the patch radius.\n -> '))
            window_radius = int(input('Enter the window radius.\n -> '))
            sigma = int(input('Enter the sigma value.\n -> '))
            _ = is_int(patch_radius, window_radius, sigma)
            _ = is_gt(patch_radius, window_radius, sigma, threshold=0)
            return NLM(
                'img/in/{}'.format(image_name or default_image),
                sigma=sigma,
                patch_radius=patch_radius,
                window_radius=window_radius,)
        except ValueError:
            print('The image name must be a valid string.'
                  'Other parameters must be positive integers.')
            return NLM.setup()

    cdef public int d, ds, D, Ds, sigma, distance_threshold
    cdef public float h

    def __init__(self, image, sigma, patch_radius, window_radius):
        """
        :param image: string path to image
        :param patch_radius: radius of patches (e.g. radius 1 gives a 3x3 patch)
        :param window_radius: radius of search window for comparison with
        reference patch
        """
        super().__init__(image)
        # Notation borrowed from IPOL article
        # Parameter-Free Fast Pixelwise Non-Local Means Denoising, pg. 304.
        # http://www.ipol.im/pub/art/2014/120/article_lr.pdf
        # Patch radius is the size of each patch.
        self.d = 2 * patch_radius + 1
        self.ds = patch_radius
        # Window radius is the search area about each pixel where patches are
        # searched.
        self.D = 2 * window_radius + 1
        self.Ds = window_radius
        # Max difference between two pixels for target patch to be included in
        # the average
        self.distance_threshold = 300
        # HYPERPARAMS
        # `h` is a parameter we set to fine tune how similar patches must be.
        self.h = .1
        # `sigma` defines how similar patches must be to be counted in the avg.
        self.sigma = sigma

    @staticmethod
    cdef pad_image(np.ndarray image, int pad_size, str mode='reflect'):
        """Pads `image` with `self.patch_radius` on each side."""
        return np.pad(image, pad_size, mode=mode)

    @staticmethod
    cdef integral_image(
            NLMIMGTYPE [:, :] padded_im,
            NLMIMGTYPE [:, ::] integral,
            int t_row,
            int t_col,
            int num_rows,
            int num_cols):
        """
        Compute the integral image from the difference image of padded_im offset
        by (t_row, t_col). Result is saved to `integral`.
        Integral image is not normalised!
        """
        # Start at 1 to stay in array bounds.
        cdef int x1, x2
        cdef float d
        for x1 in range(int_max(1, -t_row),
                        int_min(num_rows, num_rows - t_row)):
            for x2 in range(int_max(1, -t_col),
                            int_min(num_cols, num_cols - t_col)):
                # Compute diff squared
                d = (padded_im[x1, x2] - padded_im[x1 + t_row, x2 + t_col]) ** 2
                integral[x1, x2] = d \
                    + integral[x1 - 1, x2] \
                    + integral[x1, x2 - 1] \
                    - integral[x1 - 1, x2 - 1]

    @staticmethod
    cdef dist_from_integral_image(
        NLMIMGTYPE[:, ::] integral, int row, int col, int ds):
        """
        Gets the ssd of the patch centred at (row, col) in constant time using
        the given integral image.
        :param integral: integral image array
        :param row: row of the centre of the patch
        :param col: col of the centre of the patch
        """
        cdef float d
        d = integral[row + ds, col + ds] \
            - integral[row + ds, col - ds - 1] \
            - integral[row - ds - 1, col + ds] \
            + integral[row - ds - 1, col - ds - 1]
        # Returns the normalised squared difference for the patch.
        return d / (2 * ds + 1) ** 2

    def run(self, *args, **kwargs):
        # 1. INITIALISATION.
        # Set up C vars for algorithm.
        cdef int num_rows, num_cols, pad_size, shift_row, shift_col, x1, x2, row, col
        cdef float double_coefficient, d2, weight
        # Padding width on each side of the image `ds + Ds`.
        pad_size = self.ds + self.Ds
        # Set up arrays.
        cdef NLMIMGTYPE [:, :] padded_im = np.ascontiguousarray(
            NLM.pad_image(self.pixels, pad_size)).astype(np.float32)
        cdef NLMIMGTYPE [:, :] output = np.zeros_like(padded_im)
        cdef NLMIMGTYPE [:, ::] weights = np.zeros_like(
            padded_im, dtype=np.float32, order='C')
        cdef NLMIMGTYPE[:, ::] integral_diff = np.zeros_like(
            padded_im, order='C')
        # Number of rows and cols in the padded image.
        num_rows, num_cols = padded_im.shape[0], padded_im.shape[1]

        # 2. MAIN LOOP.
        # `shift_row` and `shift_col` are shifts through window area.
        for shift_row in range(-self.Ds, self.Ds + 1):
            print('{}/{}'.format(shift_row + self.Ds, self.Ds + self.Ds))
            for shift_col in range(0, self.Ds + 1):
                double_coefficient = .5 if shift_row != 0 and shift_col == 0 else 1
                integral_diff = np.zeros_like(padded_im, order='C')
                NLM.integral_image(
                    padded_im, integral_diff, shift_row, shift_col, num_rows, num_cols)
                # Iterate over every patch, accounting for shift.
                for x1 in range(int_max(self.ds, self.ds - shift_row),
                                int_min(num_rows - self.ds,
                                    num_rows - self.ds - shift_row)):
                    for x2 in range(int_max(self.ds, self.ds - shift_col),
                                    int_min(num_cols - self.ds,
                                        num_cols - self.ds - shift_col)):
                        # Get the squared difference for patch at (shift_row, shift_col)
                        # and patch at (x1, x2) using integral image. This is
                        # the measure of patch similarity.
                        d2 = NLM.dist_from_integral_image(
                            integral_diff, x1, x2, self.ds)
                        # Convert large distances to small weights.
                        # (weights of large distances are basically 0, hence the
                        # threshold).
                        if d2 > self.distance_threshold:
                            continue
                        weight = double_coefficient * exp(
                            -(max(d2 - (2 * self.sigma ** 2), 0))
                            / self.h ** 2)
                        # Add result to weight and result, exploiting the
                        # symmetry of the weight matrix (distance from ref patch
                        # and target patch are the same both ways).
                        # First, store sum of weights for later normalisation.
                        weights[x1, x2] += weight
                        weights[x1 + shift_row, x2 + shift_col] += weight
                        # Add the weighted pixel to the result. Normalised in
                        # the next step.
                        output[x1, x2] += weight * \
                            padded_im[x1 + shift_row, x2 + shift_col]
                        output[x1 + shift_row, x2 + shift_col] += weight * \
                            padded_im[x1, x2]

        # 3. COMPUTE FINAL NORMALISED ESTIMATE
        # Iterate over each pixel and divide by the sum of weights computed in
        # the main loop.
        for row in range(self.ds, num_rows - self.ds):
            for col in range(self.ds, num_cols - self.ds):
                output[row, col] /= weights[row, col]
        # Trim padding.
        cdef np.ndarray normalised_output = np.array(
            output[pad_size:-pad_size, pad_size:-pad_size], dtype=np.uint8)
        # Save the image to img/out/
        cdef str im_id = self.save(Image.fromarray(normalised_output))

        print('Done! Image ID {}'.format(im_id))
        return normalised_output
