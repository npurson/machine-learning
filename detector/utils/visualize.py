from torchvision import transforms
from PIL import ImageDraw


def visualize_result(img, bbox):
    img = transforms.ToPILImage()(img.detach().cpu())
    bbox = bbox.detach().cpu().tolist()

    draw = ImageDraw.Draw(img)
    draw.rectangle(bbox, outline=(255,0,0))
    img.save("vis.jpg")
