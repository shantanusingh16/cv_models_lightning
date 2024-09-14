import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def visualize_detection_results(images, pred_boxes, true_boxes=None, class_names=None):
    """
    Visualize detection results with side-by-side comparison if ground truth is provided.
    
    Args:
    - images: numpy array of shape (batch_size, height, width, channels)
    - pred_boxes: list of predicted bounding boxes (x1, y1, x2, y2, conf, class_id)
    - true_boxes: list of ground truth bounding boxes (x1, y1, x2, y2, class_id)
    - class_names: list of class names
    """
    batch_size = len(images)
    fig_size = (20, 10 * batch_size)
    fig, axs = plt.subplots(batch_size, 2 if true_boxes else 1, figsize=fig_size)
    
    if batch_size == 1:
        axs = np.array([axs])
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(class_names) if class_names else 1))
    
    for i in range(batch_size):
        img = images[i]
        ax_pred = axs[i, 0] if true_boxes else axs[i]
        
        ax_pred.imshow(img)
        ax_pred.set_title('Predictions')
        ax_pred.axis('off')
        
        # Plot predicted boxes
        for box in pred_boxes[i]:
            x1, y1, x2, y2, conf, class_id = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                                     edgecolor=colors[int(class_id)], facecolor='none')
            ax_pred.add_patch(rect)
            
            label = f"{class_names[int(class_id)]}: {conf:.2f}" if class_names else f"Class {int(class_id)}: {conf:.2f}"
            ax_pred.text(x1, y1, label, color=colors[int(class_id)], fontsize=8, 
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        if true_boxes:
            ax_true = axs[i, 1]
            ax_true.imshow(img)
            ax_true.set_title('Ground Truth')
            ax_true.axis('off')
            
            # Plot ground truth boxes
            for box in true_boxes[i]:
                x1, y1, x2, y2, class_id = box
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                                         edgecolor=colors[int(class_id)], facecolor='none', linestyle='--')
                ax_true.add_patch(rect)
                
                label = class_names[int(class_id)] if class_names else f"Class {int(class_id)}"
                ax_true.text(x1, y1, label, color=colors[int(class_id)], fontsize=8, 
                             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor='none', edgecolor='black', label='TP', linestyle='-'),
        patches.Patch(facecolor='none', edgecolor='black', label='FP', linestyle=':'),
        patches.Patch(facecolor='none', edgecolor='black', label='FN', linestyle='--')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3)
    
    plt.tight_layout()
    return fig

def plot_precision_recall_curve(precisions, recalls, ap):
    """
    Plot precision-recall curve.
    
    Args:
    - precisions: list of precision values
    - recalls: list of recall values
    - ap: average precision value
    """
    plt.figure(figsize=(10, 7))
    plt.plot(recalls, precisions, color='blue', label=f'Precision-Recall Curve (AP: {ap:.2f})')
    plt.fill_between(recalls, precisions, alpha=0.2, color='blue')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    return plt.gcf()

def plot_training_progress(epochs, train_losses, val_losses, train_metrics, val_metrics):
    """
    Plot training progress including loss and evaluation metrics.
    
    Args:
    - epochs: list of epoch numbers
    - train_losses: list of training losses
    - val_losses: list of validation losses
    - train_metrics: dictionary of training metrics (e.g., {'mAP': [...], 'accuracy': [...]})
    - val_metrics: dictionary of validation metrics
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    
    # Plot losses
    ax1.plot(epochs, train_losses, label='Train Loss')
    ax1.plot(epochs, val_losses, label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot metrics
    for metric in train_metrics:
        ax2.plot(epochs, train_metrics[metric], label=f'Train {metric}')
        ax2.plot(epochs, val_metrics[metric], label=f'Validation {metric}')
    
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Metric Value')
    ax2.set_title('Training and Validation Metrics')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig