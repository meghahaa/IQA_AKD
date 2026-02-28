import os
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for Kaggle/servers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class TrainingPlotter:
    """
    Tracks and plots student training metrics per epoch.

    Metrics tracked:
        - Total loss      (train, per epoch)
        - AKD loss        (train, per epoch)
        - Score loss      (train, per epoch)
        - Validation PLCC (val,   per epoch)
        - Validation SRCC (val,   per epoch)

    Usage:
        plotter = TrainingPlotter(save_dir)
        # inside epoch loop:
        plotter.update(epoch, total_loss, akd_loss, score_loss, plcc, srcc)
        plotter.save()   # overwrites the same file each epoch
    """

    def __init__(self, save_dir, filename="student_training_curves.png"):
        """
        Args:
            save_dir (str): directory to save the plot — same as checkpoint dir
            filename (str): output filename
        """
        self.save_path = os.path.join(save_dir, filename)
        os.makedirs(save_dir, exist_ok=True)

        # History — one value per epoch
        self.epochs      = []
        self.total_loss  = []
        self.akd_loss    = []
        self.score_loss  = []
        self.val_plcc    = []
        self.val_srcc    = []

    def update(self, epoch, total_loss, akd_loss, score_loss, val_plcc, val_srcc):
        """
        Call once per epoch after validation is complete.

        Args:
            epoch      (int):   current epoch number (1-indexed)
            total_loss (float): average total train loss for this epoch
            akd_loss   (float): average AKD loss for this epoch
            score_loss (float): average score loss for this epoch
            val_plcc   (float): validation PLCC
            val_srcc   (float): validation SRCC
        """
        self.epochs.append(epoch)
        self.total_loss.append(total_loss)
        self.akd_loss.append(akd_loss)
        self.score_loss.append(score_loss)
        self.val_plcc.append(val_plcc)
        self.val_srcc.append(val_srcc)

    def save(self):
        """
        Render and save the training curves figure.
        Overwrites the same file each epoch so you always have the latest.
        """
        if not self.epochs:
            return

        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor("#0f0f0f")

        gs = gridspec.GridSpec(
            2, 3,
            figure=fig,
            hspace=0.45,
            wspace=0.35,
            left=0.07, right=0.97,
            top=0.88,  bottom=0.10,
        )

        # ── Colour palette ────────────────────────────────────────────────────
        COLOR_TOTAL  = "#e05c5c"   # red    — total loss
        COLOR_AKD    = "#f0a500"   # amber  — AKD loss
        COLOR_SCORE  = "#5bc0eb"   # blue   — score loss
        COLOR_PLCC   = "#7dde92"   # green  — PLCC
        COLOR_SRCC   = "#c77dff"   # purple — SRCC
        COLOR_GRID   = "#2a2a2a"
        COLOR_TEXT   = "#e0e0e0"
        COLOR_AXES   = "#1c1c1c"

        ep = self.epochs

        def _style_ax(ax, title, ylabel, color):
            ax.set_facecolor(COLOR_AXES)
            ax.set_title(title, color=COLOR_TEXT, fontsize=11, fontweight="bold", pad=8)
            ax.set_xlabel("Epoch", color=COLOR_TEXT, fontsize=9)
            ax.set_ylabel(ylabel, color=COLOR_TEXT, fontsize=9)
            ax.tick_params(colors=COLOR_TEXT, labelsize=8)
            ax.grid(True, color=COLOR_GRID, linewidth=0.7, linestyle="--")
            for spine in ax.spines.values():
                spine.set_edgecolor(COLOR_GRID)
            ax.xaxis.set_major_locator(
                plt.MaxNLocator(integer=True, nbins=min(10, len(ep)))
            )
            # Highlight best point
            return ax

        def _plot_loss(ax, values, color, title, ylabel):
            _style_ax(ax, title, ylabel, color)
            ax.plot(ep, values, color=color, linewidth=2.0,
                    marker="o", markersize=4, zorder=3)
            ax.fill_between(ep, values, alpha=0.15, color=color)
            best_val  = min(values)
            best_ep   = ep[values.index(best_val)]
            ax.axhline(best_val, color=color, linewidth=0.8,
                       linestyle=":", alpha=0.6)
            ax.annotate(
                f"min {best_val:.4f}",
                xy=(best_ep, best_val),
                xytext=(8, 8), textcoords="offset points",
                color=color, fontsize=8,
                arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
            )

        def _plot_metric(ax, values, color, title, ylabel):
            _style_ax(ax, title, ylabel, color)
            ax.plot(ep, values, color=color, linewidth=2.0,
                    marker="o", markersize=4, zorder=3)
            ax.fill_between(ep, values, alpha=0.15, color=color)
            best_val = max(values)
            best_ep  = ep[values.index(best_val)]
            ax.axhline(best_val, color=color, linewidth=0.8,
                       linestyle=":", alpha=0.6)
            ax.annotate(
                f"max {best_val:.4f}",
                xy=(best_ep, best_val),
                xytext=(8, -14), textcoords="offset points",
                color=color, fontsize=8,
                arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
            )
            ax.set_ylim(bottom=max(0, min(values) - 0.05), top=min(1.0, max(values) + 0.05))

        # ── Row 0: loss curves ────────────────────────────────────────────────
        ax_total = fig.add_subplot(gs[0, 0])
        ax_akd   = fig.add_subplot(gs[0, 1])
        ax_score = fig.add_subplot(gs[0, 2])

        _plot_loss(ax_total, self.total_loss, COLOR_TOTAL, "Total Loss",  "Loss")
        _plot_loss(ax_akd,   self.akd_loss,   COLOR_AKD,   "AKD Loss",    "Loss")
        _plot_loss(ax_score, self.score_loss, COLOR_SCORE, "Score Loss",  "Loss")

        # ── Row 1: validation metrics (span 1.5 cols each for readability) ────
        ax_plcc = fig.add_subplot(gs[1, 0:2])
        ax_srcc = fig.add_subplot(gs[1, 2])

        _plot_metric(ax_plcc, self.val_plcc, COLOR_PLCC, "Validation PLCC", "PLCC")
        _plot_metric(ax_srcc, self.val_srcc, COLOR_SRCC, "Validation SRCC", "SRCC")

        # ── Overlay PLCC + SRCC on same axis for easy comparison ─────────────
        # Add a combined view on the PLCC subplot (it spans 2 cols)
        ax_plcc.plot(ep, self.val_srcc, color=COLOR_SRCC,
                     linewidth=1.5, linestyle="--",
                     marker="s", markersize=3, label="SRCC", zorder=2)
        ax_plcc.plot(ep, self.val_plcc, color=COLOR_PLCC,
                     linewidth=2.0, linestyle="-",
                     marker="o", markersize=4, label="PLCC", zorder=3)
        ax_plcc.legend(
            facecolor="#1c1c1c", edgecolor=COLOR_GRID,
            labelcolor=COLOR_TEXT, fontsize=8
        )
        ax_plcc.set_title("Validation PLCC & SRCC", color=COLOR_TEXT,
                          fontsize=11, fontweight="bold", pad=8)

        # ── Global title ──────────────────────────────────────────────────────
        latest_ep = ep[-1]
        fig.suptitle(
            f"Student Training Curves  —  Epoch {latest_ep}  |  "
            f"Best PLCC: {max(self.val_plcc):.4f}  |  "
            f"Best SRCC: {max(self.val_srcc):.4f}",
            color=COLOR_TEXT,
            fontsize=13,
            fontweight="bold",
            y=0.96,
        )

        plt.savefig(
            self.save_path,
            dpi=150,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)

        print(f"[Plotter] Saved training curves → {self.save_path}")