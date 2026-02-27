import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error


class IQAMetrics:
    """
    Image Quality Assessment Evaluation Metrics
    """

    @staticmethod
    def plcc(predictions, ground_truth):
        """
        Pearson Linear Correlation Coefficient (PLCC)
        Measures linear correlation between predicted and ground truth scores.

        Args:
            predictions: numpy array or tensor of predicted scores [N,]
            ground_truth: numpy array or tensor of ground truth scores [N,]

        Returns:
            plcc_value: float, correlation coefficient between -1 and 1
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(ground_truth, torch.Tensor):
            ground_truth = ground_truth.detach().cpu().numpy()

        predictions = np.asarray(predictions).flatten()
        ground_truth = np.asarray(ground_truth).flatten()

        plcc_value, _ = pearsonr(predictions, ground_truth)
        return plcc_value

    @staticmethod
    def srocc(predictions, ground_truth):
        """
        Spearman Rank Order Correlation Coefficient (SROCC)
        Measures rank-based correlation between predicted and ground truth scores.
        More robust to outliers than PLCC.

        Args:
            predictions: numpy array or tensor of predicted scores [N,]
            ground_truth: numpy array or tensor of ground truth scores [N,]

        Returns:
            srocc_value: float, Spearman correlation coefficient between -1 and 1
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(ground_truth, torch.Tensor):
            ground_truth = ground_truth.detach().cpu().numpy()

        predictions = np.asarray(predictions).flatten()
        ground_truth = np.asarray(ground_truth).flatten()

        srocc_value, _ = spearmanr(predictions, ground_truth)
        return srocc_value

    @staticmethod
    def rmse(predictions, ground_truth):
        """
        Root Mean Square Error (RMSE)
        Measures average magnitude of prediction errors.
        Lower RMSE indicates better predictions.

        Args:
            predictions: numpy array or tensor of predicted scores [N,]
            ground_truth: numpy array or tensor of ground truth scores [N,]

        Returns:
            rmse_value: float, root mean square error
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(ground_truth, torch.Tensor):
            ground_truth = ground_truth.detach().cpu().numpy()

        predictions = np.asarray(predictions).flatten()
        ground_truth = np.asarray(ground_truth).flatten()

        mse = mean_squared_error(ground_truth, predictions)
        rmse_value = np.sqrt(mse)
        return rmse_value

    @staticmethod
    def compute_all_metrics(predictions, ground_truth, verbose=True):
        """
        Compute all evaluation metrics at once.

        Args:
            predictions: numpy array or tensor of predicted scores [N,]
            ground_truth: numpy array or tensor of ground truth scores [N,]
            verbose (bool): print results if True

        Returns:
            metrics_dict: dict with keys 'plcc', 'srocc', 'rmse'
        """
        plcc_val = IQAMetrics.plcc(predictions, ground_truth)
        srocc_val = IQAMetrics.srocc(predictions, ground_truth)
        rmse_val = IQAMetrics.rmse(predictions, ground_truth)

        metrics_dict = {
            'plcc': plcc_val,
            'srocc': srocc_val,
            'rmse': rmse_val
        }

        if verbose:
            print("=" * 50)
            print("IQA Evaluation Metrics")
            print("=" * 50)
            print(f"PLCC (Pearson Correlation):  {plcc_val:.6f}")
            print(f"SROCC (Spearman Correlation): {srocc_val:.6f}")
            print(f"RMSE (Root Mean Square Error): {rmse_val:.6f}")
            print("=" * 50)

        return metrics_dict


# Convenience functions for direct usage
def compute_plcc(predictions, ground_truth):
    """Compute PLCC score."""
    return IQAMetrics.plcc(predictions, ground_truth)


def compute_srocc(predictions, ground_truth):
    """Compute SROCC score."""
    return IQAMetrics.srocc(predictions, ground_truth)


def compute_rmse(predictions, ground_truth):
    """Compute RMSE score."""
    return IQAMetrics.rmse(predictions, ground_truth)


def evaluate(predictions, ground_truth, verbose=True):
    """Compute all metrics."""
    return IQAMetrics.compute_all_metrics(predictions, ground_truth, verbose)
