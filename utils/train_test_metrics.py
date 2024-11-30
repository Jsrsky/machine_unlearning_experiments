import json
import torch
import numpy as np
from tqdm import tqdm
import seaborn as sb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from matplotlib import pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, model_name, train_loader, val_loader, criterion, optimizer, num_epochs=10):


    best_val_accuracy = 0


    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    for epoch in range(num_epochs):

        model.train()

        running_loss = 0.0

        all_preds = []
        all_labels = []

        # Training phase
        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()

            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = accuracy_score(all_labels, all_preds)

        history["train_loss"].append(epoch_loss)
        history["train_accuracy"].append(epoch_accuracy)

        # Validation phase
        model.eval()

        val_running_loss = 0.0

        val_preds = []
        val_labels = []

        with torch.inference_mode():

            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss = val_running_loss / len(val_loader.dataset)
        val_accuracy = accuracy_score(val_labels, val_preds)

        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        # Print training and validation results
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), f'{model_name}_model.pth')

    with open(f'{model_name}_history.json', 'w') as f:
        json.dump(history, f)

    print(f"Training complete for {model_name}. Training stats saved to '{model_name}_history.json'.")



def test_model(model, model_name, model_path, test_loader):
    print(f"Loading and testing model: {model_name}")

    model.load_state_dict(torch.load(model_path, weights_only=True))

    model.eval()

    test_preds = []
    test_labels = []

    with torch.inference_mode():
        for inputs, labels in tqdm(test_loader, desc="Evaluating Model"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)

            test_preds.extend(preds.cpu().numpy().tolist())
            test_labels.extend(labels.cpu().numpy().tolist())

    results = {
        "test_labels": [test_labels],
        "test_preds": [test_preds]
    }

    with open(f"{model_name}_predictions.json", "w") as f:
        json.dump(results, f)

    print(f"Predictions and labels saved to {model_name}_predictions.json")


def plot_training_history(history_path):

    with open(history_path, 'r') as f:
        data = json.load(f)


    plt.figure(figsize=(10, 5))
    plt.plot(data['train_loss'], label='Train Loss')
    plt.plot(data['val_loss'], label='Val Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


    plt.figure(figsize=(10, 5))
    plt.plot(data['train_accuracy'], label='Train Accuracy')
    plt.plot(data['val_accuracy'], label='Val Accuracy', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


def show_metrics(predictions_path, classes, model_name):

    with open(predictions_path, 'r') as f:
        data = json.load(f)


    test_labels = np.array(data['test_labels']).flatten().tolist()
    test_preds = np.array(data['test_preds']).flatten().tolist()

    # Metrics calculation
    accuracy = accuracy_score(test_labels, test_preds)
    precision = precision_score(test_labels, test_preds, average='weighted')
    recall = recall_score(test_labels, test_preds, average='weighted')
    f1 = f1_score(test_labels, test_preds, average='weighted')
    cm = confusion_matrix(test_labels, test_preds)


    # Display metrics
    print(f"Metrics for {model_name}:")
    print(f'  - Test Accuracy: {accuracy:.4f}')
    print(f'  - Precision: {precision:.4f}')
    print(f'  - Recall: {recall:.4f}')
    print(f'  - F1 Score: {f1:.4f}')

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()