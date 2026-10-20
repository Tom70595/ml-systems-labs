import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import csv
import os
sys.path.append('lab03/src')
from models import MLP, CNN, AttentionModel
from data_setup import get_dataloaders

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return running_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / total, correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128)
    criterion = nn.CrossEntropyLoss()
    
    models = {
        "MLP": MLP().to(device),
        "CNN": CNN().to(device),
        "Attention": AttentionModel().to(device)
    }
    
    epochs = 5
    lr = 1e-3
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*40}")
        print(f"Training {name}...")
        print(f"{'='*40}")
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            
        end_time = time.time()
        train_time = end_time - start_time
        
        # Test on the official 10k set
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"--> {name} Test Accuracy: {test_acc:.4f} (Trained in {train_time:.1f}s)")
        
        results[name] = {"test_acc": test_acc, "train_time": train_time}

    # Save results to CSV for the report
    os.makedirs("lab03/results", exist_ok=True)
    with open("lab03/results/training_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "test_accuracy", "train_time_seconds"])
        for name, r in results.items():
            w.writerow([name, f"{r['test_acc']:.4f}", f"{r['train_time']:.2f}"])
            
    print("\nSaved to lab03/results/training_results.csv")

if __name__ == "__main__":
    main()