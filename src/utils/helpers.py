def load_image(image_path):
    """Load an image from the specified path."""
    from PIL import Image
    return Image.open(image_path)

def preprocess_image(image, target_size):
    """Resize and normalize the image."""
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return preprocess(image)

def draw_bounding_box(image, box, color=(255, 0, 0), thickness=2):
    """Draw a bounding box on the image."""
    import cv2
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, thickness)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def calculate_iou(boxA, boxB):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def save_model(model, path):
    """Save the model to the specified path."""
    import torch
    torch.save(model.state_dict(), path)