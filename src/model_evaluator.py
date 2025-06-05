"""
Model Evaluation Module
Comprehensive model evaluation and metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self):
        self.metrics_history = []
    
    def evaluate_classification(self, y_true, y_pred, y_prob=None):
        """Comprehensive classification evaluation"""
        logger.info("Evaluating classification model...")
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        if y_prob is not None and len(np.unique(y_true)) == 2:
            # Binary classification - use probabilities for positive class
            if y_prob.ndim > 1:
                y_prob_positive = y_prob[:, 1]
            else:
                y_prob_positive = y_prob
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob_positive)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_roc_curve(self, y_true, y_prob, save_path=None):
        """Plot ROC curve"""
        if len(np.unique(y_true)) != 2:
            logger.warning("ROC curve only available for binary classification")
            return None
        
        if y_prob.ndim > 1:
            y_prob = y_prob[:, 1]
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_precision_recall_curve(self, y_true, y_prob, save_path=None):
        """Plot Precision-Recall curve"""
        if len(np.unique(y_true)) != 2:
            logger.warning("PR curve only available for binary classification")
            return None
        
        if y_prob.ndim > 1:
            y_prob = y_prob[:, 1]
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label='Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def generate_evaluation_report(self, y_true, y_pred, y_prob=None, model_name="Model"):
        """Generate comprehensive evaluation report"""
        metrics = self.evaluate_classification(y_true, y_pred, y_prob)
        
        report = f"""
        Model Evaluation Report: {model_name}
        =====================================
        
        Performance Metrics:
        - Accuracy: {metrics['accuracy']:.4f}
        - Precision: {metrics['precision']:.4f}
        - Recall: {metrics['recall']:.4f}
        - F1-Score: {metrics['f1_score']:.4f}
        """
        
        if 'roc_auc' in metrics:
            report += f"- ROC AUC: {metrics['roc_auc']:.4f}\n"
        
        report += f"""
        
        Confusion Matrix:
        {np.array(metrics['confusion_matrix'])}
        
        Detailed Classification Report:
        {classification_report(y_true, y_pred)}
        """
        
        return report

