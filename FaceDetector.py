import cv2
import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class TurtleFaceDetector:
    def __init__(self,model_path):
        BACKBONE = 'mobilenet_v2'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ENCODER_WEIGHTS = 'imagenet'
        ACTIVATION = 'sigmoid'

        # create segmentation model with pretrained encoder
        self.model = smp.Unet(
            encoder_name=BACKBONE,
            encoder_weights=ENCODER_WEIGHTS,
            classes=1,
            activation=ACTIVATION,
        )
        self.model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        self.model.to(self.device)

    def detect(self,image_path):
        # read the image
        IMAGE_SIZE = 256

        image = cv2.imread(image_path)

        # read the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # resize the image
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        # Swap color axis
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        # convert to torch tensor
        image = torch.tensor([image], dtype=torch.float)

        image = image.to(self.device, dtype=torch.float)

        output = self.model(image)

        #convert to numpy so we can stack the batches.
        output = output.cpu().detach().numpy()

        return output

if __name__ == "__main__":
    tfd = TurtleFaceDetector("0_model.bin")
    plt.figure(figsize=(20, 70))
    image = plt.imread("ninja.JPG")
    image = cv2.resize(image, (256, 256))
    pred_mask = tfd.detect("ninja.JPG")
    pred_mask = pred_mask.squeeze()
    plt.imshow(image)
    plt.imshow(pred_mask, cmap='ocean', alpha=0.3)
    plt.show()