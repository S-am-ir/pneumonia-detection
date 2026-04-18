"""
Main entry point for pneumonia detection pipeline.
Demonstrates the full workflow: data loading, training, evaluation, and visualization.
"""
import torch
from data import get_dataloaders, create_dataframe
from training import setup_model, train_one_epoch, evaluate, evaluate_on_test, visualize_gradcam
from config import NUM_EPOCHS, DATA_DIR


def main():
    """Run full training pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, test_loader, train_df = get_dataloaders()
    
    # Setup model
    model, criterion, optimizer = setup_model(train_df, device)
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{NUM_EPOCHS} - "
                  f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")
    
    # Evaluation on test set
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    evaluate_on_test(model, test_loader, criterion, device)
    
    # Visualization
    print("\n" + "="*60)
    print("GRAD-CAM VISUALIZATION")
    print("="*60)
    
    df = create_dataframe(DATA_DIR)
    visualize_gradcam(model, df, device)


if __name__ == "__main__":
    main()
