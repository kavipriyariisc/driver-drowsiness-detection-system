def load_model(model_path):
    import torch
    model = torch.load(model_path)
    model.eval()
    return model

def preprocess_input(image):
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)

def predict(model, image):
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

def main(image_path, model_path):
    from PIL import Image
    image = Image.open(image_path)
    model = load_model(model_path)
    processed_image = preprocess_input(image)
    prediction = predict(model, processed_image)
    print(f'Predicted class: {prediction}')

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python predict.py <image_path> <model_path>")
    else:
        main(sys.argv[1], sys.argv[2])