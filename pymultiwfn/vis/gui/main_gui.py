"""
Main GUI Application for PyMultiWFN

This module provides the main graphical user interface for PyMultiWFN,
replicating the functionality of the original GUI.f90 from Multiwfn.
"""

import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QSplitter, QMenuBar, QMenu, QAction,
                             QToolBar, QStatusBar, QTabWidget, QGroupBox,
                             QPushButton, QLabel, QMessageBox, QFileDialog)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QIcon, QFont

# PyMultiWFN imports
from ..molecular import MolecularVisualizer
from ..orbital import OrbitalVisualizer
from ..weak_interaction import WeakInteractionAnalyzer
from .widgets import (OrbitalSelector, ViewControls, IsosurfaceControls,
                      MoleculeControls, ProgressBar, FileSelector)

# Core imports
from ...core.data import Wavefunction
from ...io.loader import load_wavefunction


class VisualizationThread(QThread):
    """Thread for heavy visualization calculations"""
    progress_updated = pyqtSignal(int, str)
    finished_with_result = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, task_type, **kwargs):
        super().__init__()
        self.task_type = task_type
        self.kwargs = kwargs

    def run(self):
        try:
            if self.task_type == "load_file":
                self.progress_updated.emit(10, "Loading file...")
                wf = load_wavefunction(self.kwargs['filename'])
                self.progress_updated.emit(90, "Processing data...")
                self.finished_with_result.emit(wf)

            elif self.task_type == "calculate_isosurface":
                self.progress_updated.emit(10, "Generating grid...")
                # Implementation for isosurface calculation
                self.progress_updated.emit(50, "Calculating values...")
                # Continue with calculation
                self.progress_updated.emit(90, "Creating surface...")
                self.finished_with_result.emit("isosurface_data")

            elif self.task_type == "calculate_density":
                self.progress_updated.emit(10, "Generating density grid...")
                # Implementation for density calculation
                self.progress_updated.emit(90, "Finalizing...")
                self.finished_with_result.emit("density_data")

        except Exception as e:
            self.error_occurred.emit(str(e))


class MultiwfnGUI(QMainWindow):
    """
    Main GUI window for PyMultiWFN

    This class provides the main interface for visualizing molecular structures,
    orbitals, and analysis results, equivalent to the original GUI.f90 functionality.
    """

    def __init__(self):
        super().__init__()
        self.wavefunction = None
        self.current_orbital = None
        self.visualization_data = {}
        self.worker_thread = None

        self.init_ui()
        self.setup_connections()

    def init_ui(self):
        """Initialize the user interface"""
        # Window settings
        self.setWindowTitle("PyMultiWFN - Wavefunction Analysis and Visualization")
        self.setGeometry(100, 100, 1400, 900)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel - Controls
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)

        # Right panel - Visualization
        right_panel = self.create_visualization_panel()
        splitter.addWidget(right_panel)

        # Set splitter sizes (30% control, 70% visualization)
        splitter.setSizes([420, 980])

        # Create menu bar
        self.create_menu_bar()

        # Create toolbar
        self.create_toolbar()

        # Create status bar
        self.create_status_bar()

        # Initialize visualizers
        self.mol_visualizer = MolecularVisualizer()
        self.orbital_visualizer = OrbitalVisualizer()
        self.weak_interaction_analyzer = WeakInteractionAnalyzer()

    def create_control_panel(self):
        """Create the left control panel"""
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)
        control_panel.setMaximumWidth(450)

        # File selection
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout()
        self.file_selector = FileSelector(
            "Select Input File",
            "Wavefunction Files (*.wfn *.wfx *.fchk *.molden *.mwfn);;All Files (*)"
        )
        file_layout.addWidget(self.file_selector)
        file_group.setLayout(file_layout)
        control_layout.addWidget(file_group)

        # Tab widget for different control categories
        self.control_tabs = QTabWidget()

        # Orbital controls tab
        orbital_tab = QWidget()
        orbital_layout = QVBoxLayout()

        self.orbital_selector = OrbitalSelector()
        orbital_layout.addWidget(self.orbital_selector)

        self.isosurface_controls = IsosurfaceControls()
        orbital_layout.addWidget(self.isosurface_controls)

        orbital_layout.addStretch()
        orbital_tab.setLayout(orbital_layout)
        self.control_tabs.addTab(orbital_tab, "Orbitals")

        # Molecular controls tab
        molecule_tab = QWidget()
        molecule_layout = QVBoxLayout()

        self.molecule_controls = MoleculeControls()
        molecule_layout.addWidget(self.molecule_controls)

        self.view_controls = ViewControls()
        molecule_layout.addWidget(self.view_controls)

        molecule_layout.addStretch()
        molecule_tab.setLayout(molecule_layout)
        self.control_tabs.addTab(molecule_tab, "Molecule")

        # Analysis tab
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout()

        # Analysis buttons
        self.density_btn = QPushButton("Calculate Electron Density")
        self.elf_btn = QPushButton("Calculate ELF")
        self.laplacian_btn = QPushButton("Calculate Laplacian")
        self.gradient_btn = QPushButton("Calculate Gradient")
        self.weak_interaction_btn = QPushButton("Weak Interaction Analysis")

        analysis_layout.addWidget(self.density_btn)
        analysis_layout.addWidget(self.elf_btn)
        analysis_layout.addWidget(self.laplacian_btn)
        analysis_layout.addWidget(self.gradient_btn)
        analysis_layout.addWidget(self.weak_interaction_btn)

        analysis_layout.addStretch()
        analysis_tab.setLayout(analysis_layout)
        self.control_tabs.addTab(analysis_tab, "Analysis")

        control_layout.addWidget(self.control_tabs)

        # Progress bar
        self.progress_bar = ProgressBar()
        control_layout.addWidget(self.progress_bar)

        # Action buttons
        button_layout = QHBoxLayout()
        self.save_pic_btn = QPushButton("Save Picture")
        self.reset_btn = QPushButton("Reset View")
        self.export_btn = QPushButton("Export Data")

        button_layout.addWidget(self.save_pic_btn)
        button_layout.addWidget(self.reset_btn)
        button_layout.addWidget(self.export_btn)

        control_layout.addLayout(button_layout)

        return control_panel

    def create_visualization_panel(self):
        """Create the right visualization panel"""
        viz_panel = QWidget()
        viz_layout = QVBoxLayout()
        viz_panel.setLayout(viz_layout)

        # Tab widget for different visualizations
        self.viz_tabs = QTabWidget()

        # 3D molecular visualization
        self.mol_viz_widget = QWidget()
        mol_viz_layout = QVBoxLayout()

        self.mol_viz_label = QLabel("3D Molecular Visualization")
        self.mol_viz_label.setAlignment(Qt.AlignCenter)
        self.mol_viz_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ccc; min-height: 400px; }")

        mol_viz_layout.addWidget(self.mol_viz_label)
        self.mol_viz_widget.setLayout(mol_viz_layout)
        self.viz_tabs.addTab(self.mol_viz_widget, "3D Structure")

        # Orbital visualization
        self.orbital_viz_widget = QWidget()
        orbital_viz_layout = QVBoxLayout()

        self.orbital_viz_label = QLabel("Orbital Visualization")
        self.orbital_viz_label.setAlignment(Qt.AlignCenter)
        self.orbital_viz_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ccc; min-height: 400px; }")

        orbital_viz_layout.addWidget(self.orbital_viz_label)
        self.orbital_viz_widget.setLayout(orbital_viz_layout)
        self.viz_tabs.addTab(self.orbital_viz_widget, "Orbitals")

        # Weak interaction visualization
        self.weak_viz_widget = QWidget()
        weak_viz_layout = QVBoxLayout()

        self.weak_viz_label = QLabel("Weak Interaction Analysis")
        self.weak_viz_label.setAlignment(Qt.AlignCenter)
        self.weak_viz_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ccc; min-height: 400px; }")

        weak_viz_layout.addWidget(self.weak_viz_label)
        self.weak_viz_widget.setLayout(weak_viz_layout)
        self.viz_tabs.addTab(self.weak_viz_widget, "Weak Interactions")

        # 2D plots
        self.plot_viz_widget = QWidget()
        plot_viz_layout = QVBoxLayout()

        self.plot_viz_label = QLabel("2D Plots")
        self.plot_viz_label.setAlignment(Qt.AlignCenter)
        self.plot_viz_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ccc; min-height: 400px; }")

        plot_viz_layout.addWidget(self.plot_viz_label)
        self.plot_viz_widget.setLayout(plot_viz_layout)
        self.viz_tabs.addTab(self.plot_viz_widget, "2D Plots")

        viz_layout.addWidget(self.viz_tabs)

        # Information panel
        info_group = QGroupBox("System Information")
        info_layout = QVBoxLayout()

        self.info_label = QLabel("No file loaded")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("QLabel { font-family: monospace; }")

        info_layout.addWidget(self.info_label)
        info_group.setLayout(info_layout)
        viz_layout.addWidget(info_group)

        return viz_panel

    def create_menu_bar(self):
        """Create the menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('File')

        open_action = QAction('Open File', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        save_action = QAction('Save Picture', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_picture)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu('View')

        reset_view_action = QAction('Reset View', self)
        reset_view_action.triggered.connect(self.reset_view)
        view_menu.addAction(reset_view_action)

        # Tools menu
        tools_menu = menubar.addMenu('Tools')

        settings_action = QAction('Settings', self)
        settings_action.triggered.connect(self.show_settings)
        tools_menu.addAction(settings_action)

        # Help menu
        help_menu = menubar.addMenu('Help')

        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_toolbar(self):
        """Create the toolbar"""
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        open_action = QAction('Open', self)
        open_action.triggered.connect(self.open_file)
        toolbar.addAction(open_action)

        toolbar.addSeparator()

        save_action = QAction('Save', self)
        save_action.triggered.connect(self.save_picture)
        toolbar.addAction(save_action)

        toolbar.addSeparator()

        reset_action = QAction('Reset', self)
        reset_action.triggered.connect(self.reset_view)
        toolbar.addAction(reset_action)

    def create_status_bar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('Ready')

    def setup_connections(self):
        """Setup signal connections"""
        # File selector
        self.file_selector.file_selected.connect(self.load_file)

        # Orbital selector
        self.orbital_selector.orbital_selected.connect(self.on_orbital_selected)

        # Control widgets
        self.isosurface_controls.settings_changed.connect(self.on_isosurface_settings_changed)
        self.molecule_controls.settings_changed.connect(self.on_molecule_settings_changed)
        self.view_controls.view_changed.connect(self.on_view_changed)

        # Analysis buttons
        self.density_btn.clicked.connect(self.calculate_density)
        self.elf_btn.clicked.connect(self.calculate_elf)
        self.laplacian_btn.clicked.connect(self.calculate_laplacian)
        self.gradient_btn.clicked.connect(self.calculate_gradient)
        self.weak_interaction_btn.clicked.connect(self.analyze_weak_interaction)

        # Action buttons
        self.save_pic_btn.clicked.connect(self.save_picture)
        self.reset_btn.clicked.connect(self.reset_view)
        self.export_btn.clicked.connect(self.export_data)

    def load_file(self, filename):
        """Load a wavefunction file"""
        if self.worker_thread and self.worker_thread.isRunning():
            return

        self.worker_thread = VisualizationThread("load_file", filename=filename)
        self.worker_thread.progress_updated.connect(self.update_progress)
        self.worker_thread.finished_with_result.connect(self.on_file_loaded)
        self.worker_thread.error_occurred.connect(self.on_error)
        self.worker_thread.start()

        self.status_bar.showMessage(f"Loading file: {os.path.basename(filename)}")

    def on_file_loaded(self, wavefunction):
        """Handle successful file loading"""
        self.wavefunction = wavefunction
        self.update_system_info()
        self.update_orbital_list()
        self.status_bar.showMessage("File loaded successfully")
        self.progress_bar.reset()

        # Enable controls that require loaded data
        self.control_tabs.setEnabled(True)
        self.viz_tabs.setEnabled(True)

    def update_system_info(self):
        """Update system information display"""
        if self.wavefunction:
            info_text = f"Number of atoms: {len(self.wavefunction.atoms)}\n"
            info_text += f"Number of orbitals: {self.wavefunction.n_mo}\n"
            info_text += f"Basis set: {self.wavefunction.basis_set_type}\n"
            info_text += f"Charge: {self.wavefunction.charge}\n"
            info_text += f"Multiplicity: {self.wavefunction.multiplicity}"

            self.info_label.setText(info_text)

    def update_orbital_list(self):
        """Update orbital selector with loaded orbitals"""
        if self.wavefunction:
            self.orbital_selector.orbital_combo.clear()
            self.orbital_selector.orbital_combo.addItems(['None'] +
                                                         [f'{i}: {self.wavefunction.mo_energies[i-1]:.4f}'
                                                          for i in range(1, min(self.wavefunction.n_mo + 1, 100))])

    def on_orbital_selected(self, orbital_index):
        """Handle orbital selection"""
        if orbital_index >= 0 and self.wavefunction:
            self.current_orbital = orbital_index
            self.visualize_orbital(orbital_index)

    def visualize_orbital(self, orbital_index):
        """Visualize selected molecular orbital"""
        if self.worker_thread and self.worker_thread.isRunning():
            return

        self.worker_thread = VisualizationThread("calculate_isosurface",
                                                orbital_index=orbital_index,
                                                wavefunction=self.wavefunction)
        self.worker_thread.progress_updated.connect(self.update_progress)
        self.worker_thread.finished_with_result.connect(self.on_orbital_visualized)
        self.worker_thread.error_occurred.connect(self.on_error)
        self.worker_thread.start()

        self.status_bar.showMessage(f"Calculating orbital {orbital_index + 1}")

    def on_orbital_visualized(self, result):
        """Handle orbital visualization result"""
        # Update the orbital visualization widget
        self.orbital_viz_label.setText(f"Orbital {self.current_orbital + 1} visualization")
        self.status_bar.showMessage("Orbital visualization complete")
        self.progress_bar.reset()

    def calculate_density(self):
        """Calculate electron density"""
        if not self.wavefunction:
            QMessageBox.warning(self, "Warning", "Please load a wavefunction file first")
            return

        self.status_bar.showMessage("Calculating electron density...")
        # Implementation for density calculation

    def calculate_elf(self):
        """Calculate Electron Localization Function"""
        if not self.wavefunction:
            QMessageBox.warning(self, "Warning", "Please load a wavefunction file first")
            return

        self.status_bar.showMessage("Calculating ELF...")
        # Implementation for ELF calculation

    def calculate_laplacian(self):
        """Calculate Laplacian of electron density"""
        if not self.wavefunction:
            QMessageBox.warning(self, "Warning", "Please load a wavefunction file first")
            return

        self.status_bar.showMessage("Calculating Laplacian...")
        # Implementation for Laplacian calculation

    def calculate_gradient(self):
        """Calculate gradient of electron density"""
        if not self.wavefunction:
            QMessageBox.warning(self, "Warning", "Please load a wavefunction file first")
            return

        self.status_bar.showMessage("Calculating gradient...")
        # Implementation for gradient calculation

    def analyze_weak_interaction(self):
        """Perform weak interaction analysis"""
        if not self.wavefunction:
            QMessageBox.warning(self, "Warning", "Please load a wavefunction file first")
            return

        self.status_bar.showMessage("Performing weak interaction analysis...")
        # Implementation for weak interaction analysis

    def on_isosurface_settings_changed(self, settings):
        """Handle isosurface settings changes"""
        if self.current_orbital is not None:
            self.visualize_orbital(self.current_orbital)

    def on_molecule_settings_changed(self, settings):
        """Handle molecule display settings changes"""
        if self.wavefunction:
            self.visualize_molecule()

    def on_view_changed(self, view_params):
        """Handle view parameter changes"""
        # Update view for all visualizations
        pass

    def visualize_molecule(self):
        """Visualize molecular structure"""
        if self.wavefunction:
            self.mol_viz_label.setText("Molecular structure visualization")
            # Implementation for molecular visualization

    def update_progress(self, value, message):
        """Update progress bar and status"""
        self.progress_bar.set_progress(value, message)

    def on_error(self, error_message):
        """Handle errors from worker threads"""
        QMessageBox.critical(self, "Error", error_message)
        self.status_bar.showMessage("Error occurred")
        self.progress_bar.reset()

    def open_file(self):
        """Open file dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Input File", "",
            "Wavefunction Files (*.wfn *.wfx *.fchk *.molden *.mwfn);;All Files (*)"
        )
        if file_path:
            self.load_file(file_path)

    def save_picture(self):
        """Save current visualization as picture"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Picture", "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )
        if file_path:
            self.status_bar.showMessage(f"Saving picture to {os.path.basename(file_path)}")
            # Implementation for saving picture

    def reset_view(self):
        """Reset all views to default"""
        self.view_controls.reset_view()
        self.status_bar.showMessage("View reset")

    def export_data(self):
        """Export current analysis data"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Data", "",
            "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self.status_bar.showMessage(f"Exporting data to {os.path.basename(file_path)}")
            # Implementation for data export

    def show_settings(self):
        """Show settings dialog"""
        QMessageBox.information(self, "Settings", "Settings dialog not yet implemented")

    def show_about(self):
        """Show about dialog"""
        about_text = """
        PyMultiWFN Version 0.1.2

        Python-first refactor of the Multiwfn wavefunction analysis program

        This program provides comprehensive wavefunction analysis capabilities
        including molecular orbital visualization, electron density analysis,
        weak interaction analysis, and more.
        """
        QMessageBox.about(self, "About PyMultiWFN", about_text)


def main():
    """Main entry point for GUI application"""
    app = QApplication(sys.argv)
    app.setApplicationName("PyMultiWFN")
    app.setApplicationVersion("0.1.2")

    # Set application style
    app.setStyle('Fusion')

    # Create and show main window
    window = MultiwfnGUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()