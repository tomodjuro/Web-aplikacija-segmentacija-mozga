
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSlider,
    QFileDialog,
    QMessageBox,
    QCheckBox,
    QGroupBox,
)

from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)

from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, BoundaryNorm


class BraTSViewer(QMainWindow):

    MODALITIES = ["t1", "t1ce", "t2", "flair"]

    # BraTS 2020 segmentation colors
    SEG_COLORS = [
        "#FF3030",  # Label 1: NCR / NET
        "#00FF55",  # Label 2: Edema
        "#2585FF",  # Label 4: Enhancing Tumor
    ]

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "BraTS 2020 - Multimodal MRI Viewer"
        )

        self.resize(1500, 950)

        self.volumes = {}
        self.segmentation = None
        self.segmentation_path = None

        self.current_slice = 0
        self.max_slice = 0

        self.segmentation_opacity = 0.45
        self.case_name = ""

        self.setup_ui()

    # --------------------------------------------------
    # USER INTERFACE
    # --------------------------------------------------

    def setup_ui(self):

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)

        # HEADER
        header = QHBoxLayout()

        title = QLabel(
            "BraTS 2020 | Multimodal MRI Viewer"
        )

        title.setStyleSheet("""
            font-size: 22px;
            font-weight: bold;
            padding: 10px;
        """)

        self.case_label = QLabel("No case loaded")

        self.open_button = QPushButton(
            "Open BraTS Case"
        )

        self.open_button.clicked.connect(
            self.open_case
        )

        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.case_label)
        header.addWidget(self.open_button)

        main_layout.addLayout(header)

        # SEGMENTATION CONTROLS

        seg_group = QGroupBox(
            "Segmentation Overlay"
        )

        seg_layout = QHBoxLayout(seg_group)

        self.overlay_checkbox = QCheckBox(
            "Show Segmentation"
        )

        self.overlay_checkbox.setChecked(False)
        self.overlay_checkbox.setEnabled(False)

        self.overlay_checkbox.stateChanged.connect(
            self.display_images
        )

        self.opacity_label = QLabel(
            "Opacity: 45%"
        )

        self.opacity_slider = QSlider(
            Qt.Orientation.Horizontal
        )

        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(45)
        self.opacity_slider.setMaximumWidth(200)

        self.opacity_slider.valueChanged.connect(
            self.update_opacity
        )

        seg_layout.addWidget(
            self.overlay_checkbox
        )

        seg_layout.addStretch()

        seg_layout.addWidget(
            self.opacity_label
        )

        seg_layout.addWidget(
            self.opacity_slider
        )

        main_layout.addWidget(seg_group)

        # MATPLOTLIB FIGURE

        self.figure = Figure(
            facecolor="#171717"
        )

        self.canvas = FigureCanvas(
            self.figure
        )

        self.axes = self.figure.subplots(
            2, 2
        )

        self.toolbar = NavigationToolbar(
            self.canvas, self
        )

        main_layout.addWidget(
            self.toolbar
        )

        main_layout.addWidget(
            self.canvas,
            stretch=1
        )

        # SLICE CONTROLS

        controls = QHBoxLayout()

        self.prev_button = QPushButton(
            "Previous"
        )

        self.next_button = QPushButton(
            "Next"
        )

        self.prev_button.clicked.connect(
            lambda: self.change_slice(-1)
        )

        self.next_button.clicked.connect(
            lambda: self.change_slice(1)
        )

        self.slice_label = QLabel(
            "Slice: -"
        )

        self.slider = QSlider(
            Qt.Orientation.Horizontal
        )

        self.slider.setMinimum(0)
        self.slider.setMaximum(0)

        self.slider.valueChanged.connect(
            self.slider_changed
        )

        controls.addWidget(
            self.prev_button
        )

        controls.addWidget(
            self.slider,
            stretch=1
        )

        controls.addWidget(
            self.next_button
        )

        controls.addWidget(
            self.slice_label
        )

        main_layout.addLayout(controls)

        # LEGEND

        legend_layout = QHBoxLayout()

        legend_layout.addWidget(
            QLabel("Segmentation labels:")
        )

        self.add_legend_item(
            legend_layout,
            "#FF3030",
            "NCR / NET (1)"
        )

        self.add_legend_item(
            legend_layout,
            "#00FF55",
            "Edema (2)"
        )

        self.add_legend_item(
            legend_layout,
            "#2585FF",
            "Enhancing Tumor (4)"
        )

        legend_layout.addStretch()

        main_layout.addLayout(
            legend_layout
        )

        # STYLE

        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #252525;
                color: white;
            }

            QGroupBox {
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 8px;
                padding: 8px;
                font-weight: bold;
            }

            QPushButton {
                background-color: #3B78B5;
                color: white;
                padding: 9px 15px;
                border-radius: 5px;
            }

            QPushButton:hover {
                background-color: #5595D0;
            }

            QCheckBox {
                spacing: 8px;
            }

            QSlider::groove:horizontal {
                height: 6px;
                background: #555;
            }

            QSlider::handle:horizontal {
                background: #4A90E2;
                width: 15px;
                margin: -6px 0;
                border-radius: 7px;
            }
        """)

    # --------------------------------------------------
    # LEGEND
    # --------------------------------------------------

    def add_legend_item(
        self,
        layout,
        color,
        text
    ):

        color_label = QLabel("●")

        color_label.setStyleSheet(
            f"color: {color}; font-size: 18px;"
        )

        text_label = QLabel(text)

        layout.addWidget(color_label)
        layout.addWidget(text_label)

    # --------------------------------------------------
    # OPEN CASE
    # --------------------------------------------------

    def open_case(self):

        folder = QFileDialog.getExistingDirectory(
            self,
            "Select BraTS Case Folder"
        )

        if not folder:
            return

        folder = Path(folder)

        files = {}

        try:

            # Find all four modalities

            for modality in self.MODALITIES:

                matches = (
                    list(folder.glob(
                        f"*_{modality}.nii"
                    ))
                    +
                    list(folder.glob(
                        f"*_{modality}.nii.gz"
                    ))
                )

                if not matches:

                    raise FileNotFoundError(
                        f"Cannot find {modality} NIfTI file."
                    )

                if len(matches) > 1:

                    raise ValueError(
                        f"Multiple {modality} files found."
                    )

                files[modality] = matches[0]

            self.load_volumes(files)

        except Exception as error:

            QMessageBox.critical(
                self,
                "Loading error",
                str(error)
            )

    # --------------------------------------------------
    # LOAD MRI + SEGMENTATION
    # --------------------------------------------------

    def load_volumes(self, files):

        volumes = {}

        reference_shape = None
        reference_affine = None

        # LOAD FOUR MRI MODALITIES

        for modality in self.MODALITIES:

            nii = nib.load(
                str(files[modality])
            )

            nii = nib.as_closest_canonical(
                nii
            )

            data = nii.get_fdata(
                dtype=np.float32
            )

            if data.ndim != 3:

                raise ValueError(
                    f"{modality} must be a 3D volume."
                )

            if not np.all(np.isfinite(data)):

                raise ValueError(
                    f"{modality} contains invalid values."
                )

            if reference_shape is None:

                reference_shape = data.shape
                reference_affine = nii.affine

            else:

                if data.shape != reference_shape:

                    raise ValueError(
                        "MRI volume dimensions differ."
                    )

                if not np.allclose(
                    nii.affine,
                    reference_affine,
                    atol=1e-3
                ):

                    raise ValueError(
                        "MRI affine transformations differ."
                    )

            volumes[modality] = data

        # LOAD SEGMENTATION

        segmentation = None
        segmentation_path = None

        folder = files["t1"].parent

        seg_files = (
            list(folder.glob("*_seg.nii"))
            +
            list(folder.glob("*_seg.nii.gz"))
        )

        if len(seg_files) > 1:

            raise ValueError(
                "Multiple segmentation files found."
            )

        if seg_files:

            seg_nii = nib.load(
                str(seg_files[0])
            )

            seg_nii = nib.as_closest_canonical(
                seg_nii
            )

            seg_data = seg_nii.get_fdata(
                dtype=np.float32
            )

            if seg_data.ndim != 3:

                raise ValueError(
                    "Segmentation must be a 3D volume."
                )

            if seg_data.shape != reference_shape:

                raise ValueError(
                    "Segmentation dimensions do not match MRI."
                )

            if not np.allclose(
                seg_nii.affine,
                reference_affine,
                atol=1e-3
            ):

                raise ValueError(
                    "Segmentation affine does not match MRI."
                )

            if not np.all(np.isfinite(seg_data)):

                raise ValueError(
                    "Segmentation contains invalid values."
                )

            labels = np.unique(seg_data)

            if not np.all(
                np.isin(labels, [0, 1, 2, 4])
            ):

                raise ValueError(
                    f"Unexpected segmentation labels: {labels}"
                )

            segmentation = seg_data.astype(
                np.uint8
            )

            segmentation_path = seg_files[0]

        # ASSIGN DATA AFTER SUCCESSFUL VALIDATION

        self.volumes = volumes

        self.segmentation = segmentation
        self.segmentation_path = segmentation_path

        self.max_slice = (
            reference_shape[2] - 1
        )

        self.current_slice = (
            reference_shape[2] // 2
        )

        self.case_name = folder.name

        self.slider.blockSignals(True)

        self.slider.setRange(
            0,
            self.max_slice
        )

        self.slider.setValue(
            self.current_slice
        )

        self.slider.blockSignals(False)

        self.case_label.setText(
            self.case_name
        )

        self.overlay_checkbox.setEnabled(
            segmentation is not None
        )

        self.overlay_checkbox.setChecked(
            segmentation is not None
        )

        if segmentation is None:

            QMessageBox.information(
                self,
                "Segmentation",
                "No segmentation file found. "
                "MRI modalities loaded successfully."
            )

        self.display_images()

    # --------------------------------------------------
    # DISPLAY MRI + OVERLAY
    # --------------------------------------------------

    def display_images(self, *args):

        if not self.volumes:
            return

        for ax in self.axes.flat:

            ax.clear()

        for ax, modality in zip(
            self.axes.flat,
            self.MODALITIES
        ):

            volume = self.volumes[modality]

            image = volume[
                :, :, self.current_slice
            ]

            # MRI INTENSITY SCALING

            low, high = np.percentile(
                image,
                [1, 99]
            )

            if high <= low:
                high = low + 1

            ax.imshow(
                image.T,
                cmap="gray",
                origin="lower",
                vmin=low,
                vmax=high,
                interpolation="nearest"
            )

            # SEGMENTATION OVERLAY

            if (
                self.segmentation is not None
                and self.overlay_checkbox.isChecked()
            ):

                seg_slice = self.segmentation[
                    :, :, self.current_slice
                ].T

                masked_seg = np.ma.masked_where(
                    seg_slice == 0,
                    seg_slice
                )

                cmap = ListedColormap(
                    self.SEG_COLORS
                )

                norm = BoundaryNorm(
                    [0.5, 1.5, 3.0, 4.5],
                    cmap.N
                )

                ax.imshow(
                    masked_seg,
                    cmap=cmap,
                    norm=norm,
                    alpha=self.segmentation_opacity,
                    origin="lower",
                    interpolation="nearest"
                )

            ax.set_title(
                modality.upper(),
                color="white",
                fontsize=15,
                fontweight="bold"
            )

            ax.axis("off")

        self.figure.subplots_adjust(
            left=0.02,
            right=0.98,
            bottom=0.02,
            top=0.95,
            wspace=0.04,
            hspace=0.12
        )

        self.slice_label.setText(
            f"Slice: {self.current_slice} / {self.max_slice}"
        )

        self.canvas.draw_idle()

    # --------------------------------------------------
    # OPACITY
    # --------------------------------------------------

    def update_opacity(self, value):

        self.segmentation_opacity = (
            value / 100.0
        )

        self.opacity_label.setText(
            f"Opacity: {value}%"
        )

        self.display_images()

    # --------------------------------------------------
    # SLICE NAVIGATION
    # --------------------------------------------------

    def slider_changed(self, value):

        self.current_slice = value

        self.display_images()

    def change_slice(self, step):

        if not self.volumes:
            return

        new_slice = np.clip(
            self.current_slice + step,
            0,
            self.max_slice
        )

        self.slider.setValue(
            int(new_slice)
        )


# ------------------------------------------------------
# MAIN
# ------------------------------------------------------

if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = BraTSViewer()

    window.show()

    sys.exit(app.exec())
