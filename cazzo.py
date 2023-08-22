from utils.constants import *
from utils.utils import *
import utils.utils as utils



def main():
    x = np.random.random(3)

    # By default, it will use IMAGENET constants
    processed_image1 = utils.pre_process_numpy_img(x)
    print(processed_image1)
    # Switch to CLIP constants
    ConstantsContext.use_clip()
    processed_image2 = utils.pre_process_numpy_img(x)
    print(processed_image2)


if __name__ == "__main__":
    main()
    
