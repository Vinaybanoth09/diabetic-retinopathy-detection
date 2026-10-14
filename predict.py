from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_DIR / 'checkpoints' / 'efficientnet_b0_best.pth'
GRADE_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

def load_model(checkpoint_path=DEFAULT_CHECKPOINT, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    return model, device

def predict_image(image_path, model, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    with Image.open(image_path) as image:
        tensor = transform(image.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        scores = torch.softmax(model(tensor), dim=1)[0]
    predicted = int(scores.argmax().item())
    return {
        'predicted_grade': predicted,
        'predicted_label': GRADE_NAMES[predicted],
        'softmax_score_percent': round(float(scores[predicted]) * 100, 2),
        'class_scores_percent': {GRADE_NAMES[i]: round(float(scores[i]) * 100, 2)
                                 for i in range(5)},
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Predict DR grade from a retinal image.')
    parser.add_argument('--image', required=True, help='Path to retinal fundus image')
    parser.add_argument('--checkpoint', default=str(DEFAULT_CHECKPOINT),
                        help='Optional path to model checkpoint')
    args = parser.parse_args()
    model, device = load_model(args.checkpoint)
    print(predict_image(args.image, model, device))
