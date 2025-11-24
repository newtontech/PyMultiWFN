"""
GUI Widgets for PyMultiWFN

This module contains various GUI widgets used in the main interface.
"""

import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSlider,
                             QLabel, QSpinBox, QDoubleSpinBox, QPushButton,
                             QComboBox, QCheckBox, QGroupBox, QGridLayout,
                             QProgressBar, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont


class OrbitalSelector(QWidget):
    """Widget for selecting molecular orbitals"""

    orbital_selected = pyqtSignal(int)

    def __init__(self, n_mo=100):
        super().__init__()
        self.n_mo = n_mo
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Orbital list
        self.orbital_combo = QComboBox()
        self.orbital_combo.addItems(['None'] + [f'{i}' for i in range(1, self.n_mo + 1)])
        self.orbital_combo.currentIndexChanged.connect(self.on_orbital_selected)

        # Orbital info display
        self.info_label = QLabel("Select an orbital")
        self.info_label.setWordWrap(True)

        layout.addWidget(QLabel("Orbital:"))
        layout.addWidget(self.orbital_combo)
        layout.addWidget(self.info_label)

        self.setLayout(layout)

    def on_orbital_selected(self, index):
        if index > 0:  # Skip "None" option
            self.orbital_selected.emit(index - 1)  # Convert to 0-based
        else:
            self.orbital_selected.emit(-1)  # No orbital selected

    def set_orbital_info(self, info_text):
        """Update orbital information display"""
        self.info_label.setText(info_text)


class ViewControls(QWidget):
    """Widget for controlling 3D view parameters"""

    view_changed = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QGroupBox("View Controls")
        main_layout = QVBoxLayout()

        # Rotation controls
        rot_layout = QGridLayout()
        self.x_rot_slider = self.create_slider(-180, 180, 0, "X Rotation:")
        self.y_rot_slider = self.create_slider(-180, 180, 0, "Y Rotation:")
        self.z_rot_slider = self.create_slider(-180, 180, 0, "Z Rotation:")

        rot_layout.addWidget(QLabel("X Rotation:"), 0, 0)
        rot_layout.addWidget(self.x_rot_slider, 0, 1)
        rot_layout.addWidget(QLabel("Y Rotation:"), 1, 0)
        rot_layout.addWidget(self.y_rot_slider, 1, 1)
        rot_layout.addWidget(QLabel("Z Rotation:"), 2, 0)
        rot_layout.addWidget(self.z_rot_slider, 2, 1)

        # Zoom control
        zoom_layout = QHBoxLayout()
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(1, 200)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.on_view_changed)

        zoom_layout.addWidget(QLabel("Zoom:"))
        zoom_layout.addWidget(self.zoom_slider)

        # Buttons
        button_layout = QHBoxLayout()
        self.reset_btn = QPushButton("Reset View")
        self.reset_btn.clicked.connect(self.reset_view)

        button_layout.addWidget(self.reset_btn)
        button_layout.addStretch()

        main_layout.addLayout(rot_layout)
        main_layout.addLayout(zoom_layout)
        main_layout.addLayout(button_layout)
        layout.setLayout(main_layout)

        self.setLayout(layout)

    def create_slider(self, min_val, max_val, default, label):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        slider.valueChanged.connect(self.on_view_changed)
        return slider

    def on_view_changed(self):
        """Emit view change signal"""
        view_params = {
            'x_rot': self.x_rot_slider.value(),
            'y_rot': self.y_rot_slider.value(),
            'z_rot': self.z_rot_slider.value(),
            'zoom': self.zoom_slider.value() / 100.0
        }
        self.view_changed.emit(view_params)

    def reset_view(self):
        """Reset view to default"""
        self.x_rot_slider.setValue(0)
        self.y_rot_slider.setValue(0)
        self.z_rot_slider.setValue(0)
        self.zoom_slider.setValue(100)


class IsosurfaceControls(QWidget):
    """Widget for controlling isosurface visualization"""

    settings_changed = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QGroupBox("Isosurface Settings")
        main_layout = QVBoxLayout()

        # Isovalue control
        iso_layout = QHBoxLayout()
        self.iso_value_spin = QDoubleSpinBox()
        self.iso_value_spin.setRange(-1.0, 1.0)
        self.iso_value_spin.setSingleStep(0.001)
        self.iso_value_spin.setDecimals(4)
        self.iso_value_spin.setValue(0.02)
        self.iso_value_spin.valueChanged.connect(self.on_settings_changed)

        iso_layout.addWidget(QLabel("Isovalue:"))
        iso_layout.addWidget(self.iso_value_spin)

        # Quality control
        quality_layout = QHBoxLayout()
        self.quality_combo = QComboBox()
        self.quality_combo.addItems([
            "Very Poor (25k points)",
            "Poor (50k points)",
            "Default (120k points)",
            "Good (300k points)",
            "High (500k points)",
            "Very High (1000k points)",
            "Perfect (1500k points)"
        ])
        self.quality_combo.setCurrentIndex(2)  # Default
        self.quality_combo.currentIndexChanged.connect(self.on_settings_changed)

        quality_layout.addWidget(QLabel("Quality:"))
        quality_layout.addWidget(self.quality_combo)

        # Style control
        style_layout = QHBoxLayout()
        self.style_combo = QComboBox()
        self.style_combo.addItems([
            "Solid Face",
            "Mesh",
            "Points",
            "Solid + Mesh",
            "Transparent Face"
        ])
        self.style_combo.currentIndexChanged.connect(self.on_settings_changed)

        style_layout.addWidget(QLabel("Style:"))
        style_layout.addWidget(self.style_combo)

        # Color controls
        color_layout = QHBoxLayout()
        self.pos_color_btn = QPushButton("Positive Color")
        self.neg_color_btn = QPushButton("Negative Color")

        color_layout.addWidget(self.pos_color_btn)
        color_layout.addWidget(self.neg_color_btn)

        main_layout.addLayout(iso_layout)
        main_layout.addLayout(quality_layout)
        main_layout.addLayout(style_layout)
        main_layout.addLayout(color_layout)

        layout.setLayout(main_layout)
        self.setLayout(layout)

    def on_settings_changed(self):
        """Emit settings change signal"""
        settings = {
            'isovalue': self.iso_value_spin.value(),
            'quality': self.quality_combo.currentIndex(),
            'style': self.style_combo.currentIndex()
        }
        self.settings_changed.emit(settings)


class MoleculeControls(QWidget):
    """Widget for controlling molecular structure display"""

    settings_changed = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QGroupBox("Molecule Settings")
        main_layout = QVBoxLayout()

        # Atomic size control
        size_layout = QHBoxLayout()
        self.atom_size_spin = QDoubleSpinBox()
        self.atom_size_spin.setRange(0.1, 5.0)
        self.atom_size_spin.setSingleStep(0.1)
        self.atom_size_spin.setValue(1.0)
        self.atom_size_spin.valueChanged.connect(self.on_settings_changed)

        size_layout.addWidget(QLabel("Atomic Size:"))
        size_layout.addWidget(self.atom_size_spin)

        # Bond threshold
        bond_layout = QHBoxLayout()
        self.bond_threshold_spin = QDoubleSpinBox()
        self.bond_threshold_spin.setRange(0.0, 5.0)
        self.bond_threshold_spin.setSingleStep(0.05)
        self.bond_threshold_spin.setValue(1.15)
        self.bond_threshold_spin.valueChanged.connect(self.on_settings_changed)

        bond_layout.addWidget(QLabel("Bond Threshold:"))
        bond_layout.addWidget(self.bond_threshold_spin)

        # Display options
        self.show_labels_cb = QCheckBox("Show Atomic Labels")
        self.show_labels_cb.setChecked(True)
        self.show_labels_cb.toggled.connect(self.on_settings_changed)

        self.show_axis_cb = QCheckBox("Show Axis")
        self.show_axis_cb.setChecked(True)
        self.show_axis_cb.toggled.connect(self.on_settings_changed)

        self.show_hydrogens_cb = QCheckBox("Show Hydrogens")
        self.show_hydrogens_cb.setChecked(True)
        self.show_hydrogens_cb.toggled.connect(self.on_settings_changed)

        # Atomic style
        style_layout = QHBoxLayout()
        self.style_combo = QComboBox()
        self.style_combo.addItems(["CPK", "VDW", "Line"])
        self.style_combo.currentIndexChanged.connect(self.on_settings_changed)

        style_layout.addWidget(QLabel("Atomic Style:"))
        style_layout.addWidget(self.style_combo)

        main_layout.addLayout(size_layout)
        main_layout.addLayout(bond_layout)
        main_layout.addLayout(style_layout)
        main_layout.addWidget(self.show_labels_cb)
        main_layout.addWidget(self.show_axis_cb)
        main_layout.addWidget(self.show_hydrogens_cb)

        layout.setLayout(main_layout)
        self.setLayout(layout)

    def on_settings_changed(self):
        """Emit settings change signal"""
        settings = {
            'atom_size': self.atom_size_spin.value(),
            'bond_threshold': self.bond_threshold_spin.value(),
            'style': self.style_combo.currentIndex(),
            'show_labels': self.show_labels_cb.isChecked(),
            'show_axis': self.show_axis_cb.isChecked(),
            'show_hydrogens': self.show_hydrogens_cb.isChecked()
        }
        self.settings_changed.emit(settings)


class ProgressBar(QWidget):
    """Widget for showing progress of calculations"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.status_label = QLabel("Ready")

        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def set_progress(self, value, status_text=None):
        """Update progress bar and status"""
        self.progress_bar.setValue(value)
        if status_text:
            self.status_label.setText(status_text)

    def reset(self):
        """Reset progress bar"""
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready")


class FileSelector(QWidget):
    """Widget for file selection"""

    file_selected = pyqtSignal(str)

    def __init__(self, title="Select File", file_filter="All Files (*)"):
        super().__init__()
        self.title = title
        self.file_filter = file_filter
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()

        self.path_label = QLabel("No file selected")
        self.path_label.setWordWrap(True)

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_file)

        layout.addWidget(self.path_label)
        layout.addWidget(self.browse_btn)

        self.setLayout(layout)

    def browse_file(self):
        """Open file dialog"""
        file_path, _ = QFileDialog.getOpenFileName(self, self.title, "", self.file_filter)
        if file_path:
            self.path_label.setText(file_path)
            self.file_selected.emit(file_path)

    def get_file_path(self):
        """Get current file path"""
        return self.path_label.text() if self.path_label.text() != "No file selected" else None