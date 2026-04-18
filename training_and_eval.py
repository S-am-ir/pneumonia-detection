"""
training.py
Model setup, training loop, evaluation, and visualization.
"""
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from config import LEARNING_RATE, MODEL_CHECKPOINT


# ============================================================================
# MODEL SETUP
# ============================================================================

def setup_model(train_df, device):
    """Initialize EfficientNet-B0 model with weighted loss."""
    # Load pretrained EfficientNet-B0
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Modify classifier for binary output
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Linear(num_ftrs, 2)  # Normal (0) vs Pneumonia (1)
    
    model = model.to(device)
    
    # Class weights for imbalance handling
    class_counts = train_df['label'].value_counts().sort_index().values
    class_weights = torch.tensor([1.0/class_counts[0], 1.0/class_counts[1]], 
                                dtype=torch.float32).to(device)
    class_weights = class_weights / class_weights.sum() * 2.0
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("✅ Model initialized on", device)
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    return model, criterion, optimizer


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch and return metrics."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return total_loss / len(loader), acc, prec, rec, f1


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(model, loader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return total_loss / len(loader), acc, prec, rec, f1, all_preds, all_labels


def evaluate_on_test(model, test_loader, criterion, device):
    """Load best model and evaluate on test set."""
    model.load_state_dict(torch.load(MODEL_CHECKPOINT, weights_only=True))
    model.eval()
    
    test_loss, test_acc, test_prec, test_rec, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    
    print("\n" + "="*60)
    print("FINAL RESULTS ON TEST SET (624 images)")
    print("="*60)
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Accuracy      : {test_acc:.4f}")
    print(f"Precision     : {test_prec:.4f}")
    print(f"Recall        : {test_rec:.4f}   ← Very important (catching Pneumonia)")
    print(f"F1-Score      : {test_f1:.4f}")
    print("="*60)
    
    return test_loss, test_acc, test_prec, test_rec, test_f1, test_preds, test_labels


# ============================================================================
# GRAD-CAM VISUALIZATION
# ============================================================================

def visualize_gradcam(model, test_df, device, target_idx=7):
    """Generate and display Grad-CAM visualization."""
    from data import get_transforms
    
    target_layers = [model.features[-1]]
    
    idx = target_idx
    img_path = test_df.iloc[idx]['image_path']
    label = test_df.iloc[idx]['label']
    label_name = test_df.iloc[idx]['label_name']
    
    class_name = "PNEUMONIA" if label == 1 else "NORMAL"
    
    # Load and prepare image
    image_pil = Image.open(img_path).convert('RGB')
    rgb_img = np.array(image_pil.resize((224, 224))) / 255.0
    
    # Prepare input tensor
    _, val_transform = get_transforms()
    transformed = val_transform(image=np.array(image_pil))
    input_tensor = transformed['image'].unsqueeze(0)
    input_tensor = input_tensor.to(device)
    
    # Generate Grad-CAM
    with EigenCAM(model=model, target_layers=target_layers) as cam:
        cam_map = cam(input_tensor=input_tensor,
                      targets=[ClassifierOutputTarget(label)],
                      aug_smooth=True,
                      eigen_smooth=True)[0]
    
    # Create visualization
    visualization = show_cam_on_image(rgb_img, cam_map, use_rgb=True)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output[0], dim=0)
        pred_label = torch.argmax(output[0]).item()
        confidence = probs[pred_label].item() * 100
    
    pred_class = "PNEUMONIA" if pred_label == 1 else "NORMAL"
    true_class = "PNEUMONIA" if label == 1 else "NORMAL"
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.title(f"Original — True: {true_class}", fontsize=13, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title(f"Grad-CAM\nPredicted: {pred_class} ({confidence:.1f}% confidence)", 
              fontsize=13, fontweight='bold')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
