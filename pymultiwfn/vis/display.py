"""
Display and Plotting Module for PyMultiWFN

This module provides comprehensive plotting and display functionality,
including 2D plots, contour plots, scatter plots, and specialized visualizations.
Based on the plotting functionality from Multiwfn's dislin_d module.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

from ..core.data import Wavefunction
from ..config import Config


class PlotStyle(Enum):
    """Available plot styles"""
    DEFAULT = "default"
    SEABORN = "seaborn-v0_8"
    CLASSIC = "classic"
    MODERN = "seaborn-v0_8-whitegrid"


class GraphFormat(Enum):
    """Available graph output formats"""
    PNG = "png"
    SVG = "svg"
    PDF = "pdf"
    JPG = "jpg"


@dataclass
class PlotSettings:
    """Settings for plot generation"""
    style: PlotStyle = PlotStyle.DEFAULT
    figure_size: Tuple[int, int] = (10, 8)
    dpi: int = 300
    color_map: str = "viridis"
    font_size: int = 12
    line_width: float = 2.0
    marker_size: float = 6.0
    alpha: float = 0.8


class Plotter:
    """
    Advanced plotting class for PyMultiWFN

    Provides comprehensive 2D and 3D plotting capabilities for various types of data,
    including specialized plots for quantum chemical analysis based on Multiwfn functionality.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the plotter

        Parameters
        ----------
        config : Config, optional
            Configuration object for plot settings
        """
        self.config = config or Config()
        self.settings = PlotSettings()

        # Set plot style
        try:
            plt.style.use(self.settings.style.value)
        except:
            plt.style.use('default')

        # Color maps for different types of data
        self.color_maps = {
            'density': 'viridis',
            'orbital': 'RdBu_r',
            'gradient': 'plasma',
            'potential': 'coolwarm',
            'rdg': 'hot',
            'sign_lambda2': 'seismic'
        }

    def plot_1d(self, x: np.ndarray, y: np.ndarray,
                xlabel: str = 'X', ylabel: str = 'Y',
                title: str = 'Plot', color: str = 'blue') -> plt.Figure:
        """Create a 1D plot"""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x, y, color=color, linewidth=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return fig

    def plot_contour(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                     xlabel: str = 'X', ylabel: str = 'Y',
                     title: str = 'Contour Plot') -> plt.Figure:
        """Create a contour plot"""
        fig, ax = plt.subplots(figsize=(8, 6))
        contour = ax.contourf(x, y, z, levels=20, cmap='viridis')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        plt.colorbar(contour, ax=ax)
        return fig

    def plot_3d_surface(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                       xlabel: str = 'X', ylabel: str = 'Y', zlabel: str = 'Z',
                       title: str = '3D Surface Plot', colorscale: str = 'Viridis') -> go.Figure:
        """Create a 3D surface plot using plotly"""
        fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, colorscale=colorscale)])
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                zaxis_title=zlabel
            )
        )
        return fig

    def plot_scatter(self, x: np.ndarray, y: np.ndarray,
                    xlabel: str = 'X', ylabel: str = 'Y',
                    title: str = 'Scatter Plot',
                    xlabel_tex: str = None, ylabel_tex: str = None,
                    color: str = 'blue', alpha: float = 0.6,
                    xlim: Tuple[float, float] = None,
                    ylim: Tuple[float, float] = None) -> plt.Figure:
        """
        Create a scatter plot, commonly used for NCI/IGM analysis

        Parameters
        ----------
        x, y : np.ndarray
            Data arrays for x and y axes
        xlabel, ylabel : str
            Axis labels
        title : str
            Plot title
        xlabel_tex, ylabel_tex : str, optional
            TeX-formatted labels for scientific notation
        color : str
            Marker color
        alpha : float
            Marker transparency
        xlim, ylim : tuple, optional
            Axis limits

        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.settings.figure_size)

        # Filter data points if limits are specified
        if xlim is not None:
            mask = (x >= xlim[0]) & (x <= xlim[1])
            x, y = x[mask], y[mask]
        if ylim is not None:
            mask = (y >= ylim[0]) & (y <= ylim[1])
            x, y = x[mask], y[mask]

        # Create scatter plot with density coloring
        scatter = ax.scatter(x, y, c=color, alpha=alpha, s=1, rasterized=True)

        # Set labels with TeX support if provided
        if xlabel_tex:
            ax.set_xlabel(xlabel_tex, fontsize=self.settings.font_size)
        else:
            ax.set_xlabel(xlabel, fontsize=self.settings.font_size)

        if ylabel_tex:
            ax.set_ylabel(ylabel_tex, fontsize=self.settings.font_size)
        else:
            ax.set_ylabel(ylabel, fontsize=self.settings.font_size)

        ax.set_title(title, fontsize=self.settings.font_size + 2, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Set limits
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        plt.tight_layout()
        return fig

    def plot_nci_scatter(self, sign_lambda2_rho: np.ndarray, rdg: np.ndarray,
                        xlim: Tuple[float, float] = (-0.5, 0.5),
                        ylim: Tuple[float, float] = (0, 2),
                        title: str = 'NCI Analysis: sign(λ₂)ρ vs RDG') -> plt.Figure:
        """
        Create NCI scatter plot

        Parameters
        ----------
        sign_lambda2_rho : np.ndarray
            Sign(λ₂)ρ values
        rdg : np.ndarray
            Reduced density gradient values
        xlim, ylim : tuple
            Plot limits
        title : str
            Plot title

        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        return self.plot_scatter(
            sign_lambda2_rho, rdg,
            xlabel='sign(λ₂)ρ (a.u.)',
            ylabel='Reduced Density Gradient',
            xlabel_tex=r'sign($\lambda_2$)\rho$ (a.u.)',
            ylabel_tex='RDG',
            title=title,
            color='blue',
            xlim=xlim,
            ylim=ylim
        )

    def plot_igm_scatter(self, sign_lambda2_rho: np.ndarray, delta_g: np.ndarray,
                       plot_type: str = 'total',
                       xlim: Tuple[float, float] = (-0.5, 0.5),
                       ylim: Tuple[float, float] = None) -> plt.Figure:
        """
        Create IGM scatter plot

        Parameters
        ----------
        sign_lambda2_rho : np.ndarray
            Sign(λ₂)ρ values
        delta_g : np.ndarray or tuple
            δg values or (δg_inter, δg_intra, δg_total)
        plot_type : str
            Type of plot: 'inter', 'intra', 'total', or 'combined'
        xlim, ylim : tuple
            Plot limits

        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        if isinstance(delta_g, tuple) and len(delta_g) == 3:
            delta_g_inter, delta_g_intra, delta_g_total = delta_g
        else:
            delta_g_total = delta_g
            delta_g_inter = delta_g_intra = None

        if plot_type == 'inter' and delta_g_inter is not None:
            y_data = delta_g_inter
            ylabel = r'δg$^{inter}$ (a.u.)'
            title = 'IGM Analysis: δg_inter vs sign(λ₂)ρ'
        elif plot_type == 'intra' and delta_g_intra is not None:
            y_data = delta_g_intra
            ylabel = r'δg$^{intra}$ (a.u.)'
            title = 'IGM Analysis: δg_intra vs sign(λ₂)ρ'
        else:
            y_data = delta_g_total
            ylabel = r'δg$ (a.u.)'
            title = 'IGM Analysis: δg vs sign(λ₂)ρ'

        if ylim is None:
            ylim = (0, np.percentile(y_data[y_data > 0], 99))

        return self.plot_scatter(
            sign_lambda2_rho, y_data,
            xlabel='sign(λ₂)ρ (a.u.)',
            ylabel=ylabel,
            xlabel_tex=r'sign($\lambda_2$)\rho$ (a.u.)',
            ylabel_tex=ylabel,
            title=title,
            color='red',
            xlim=xlim,
            ylim=ylim
        )

    def plot_density_map(self, x: np.ndarray, y: np.ndarray, density: np.ndarray,
                         xlabel: str = 'X', ylabel: str = 'Y',
                         title: str = 'Electron Density Map',
                         levels: int = 20, colorscale: str = 'viridis') -> plt.Figure:
        """
        Create 2D density map

        Parameters
        ----------
        x, y : np.ndarray
            Grid coordinates
        density : np.ndarray
            Density values on grid
        xlabel, ylabel : str
            Axis labels
        title : str
            Plot title
        levels : int
            Number of contour levels
        colorscale : str
            Colormap name

        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.settings.figure_size)

        # Create contour plot
        if len(density.shape) == 1:
            # Reshape 1D data to 2D grid
            nx = int(np.sqrt(len(density)))
            ny = len(density) // nx
            density_2d = density[:nx*ny].reshape(nx, ny)
        else:
            density_2d = density

        # Use appropriate colormap
        cmap = self.color_maps.get('density', colorscale)

        contour = ax.contourf(x[:density_2d.shape[0]], y[:density_2d.shape[1]],
                             density_2d, levels=levels, cmap=cmap)
        ax.set_xlabel(xlabel, fontsize=self.settings.font_size)
        ax.set_ylabel(ylabel, fontsize=self.settings.font_size)
        ax.set_title(title, fontsize=self.settings.font_size + 2, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Density (a.u.)', fontsize=self.settings.font_size)

        plt.tight_layout()
        return fig

    def plot_potential_map(self, x: np.ndarray, y: np.ndarray, potential: np.ndarray,
                          xlabel: str = 'X', ylabel: str = 'Y',
                          title: str = 'Potential Map',
                          levels: int = 20) -> plt.Figure:
        """
        Create 2D potential map (e.g., electrostatic, vdW potential)

        Parameters
        ----------
        x, y : np.ndarray
            Grid coordinates
        potential : np.ndarray
            Potential values on grid
        xlabel, ylabel : str
            Axis labels
        title : str
            Plot title
        levels : int
            Number of contour levels

        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.settings.figure_size)

        # Create contour plot with divergent colormap
        if len(potential.shape) == 1:
            # Reshape 1D data to 2D grid
            nx = int(np.sqrt(len(potential)))
            ny = len(potential) // nx
            potential_2d = potential[:nx*ny].reshape(nx, ny)
        else:
            potential_2d = potential

        # Use divergent colormap for potential
        cmap = self.color_maps.get('potential', 'RdBu_r')

        # Create symmetric levels for better visualization
        vmax = np.max(np.abs(potential_2d))
        levels_array = np.linspace(-vmax, vmax, levels)

        contour = ax.contourf(x[:potential_2d.shape[0]], y[:potential_2d.shape[1]],
                             potential_2d, levels=levels_array, cmap=cmap)
        ax.set_xlabel(xlabel, fontsize=self.settings.font_size)
        ax.set_ylabel(ylabel, fontsize=self.settings.font_size)
        ax.set_title(title, fontsize=self.settings.font_size + 2, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Potential (a.u.)', fontsize=self.settings.font_size)

        plt.tight_layout()
        return fig

    def save_plot(self, fig: plt.Figure, filename: str,
                  format: GraphFormat = GraphFormat.PNG,
                  dpi: int = None) -> None:
        """
        Save plot to file

        Parameters
        ----------
        fig : plt.Figure
            Matplotlib figure to save
        filename : str
            Output filename
        format : GraphFormat
            Output format
        dpi : int, optional
            Resolution for raster formats
        """
        if dpi is None:
            dpi = self.settings.dpi

        fig.savefig(filename, format=format.value, dpi=dpi, bbox_inches='tight')

    def create_multi_panel_figure(self, plots: List[plt.Figure],
                                 titles: List[str],
                                 layout: Tuple[int, int] = None) -> plt.Figure:
        """
        Create multi-panel figure from multiple plots

        Parameters
        ----------
        plots : List[plt.Figure]
            List of individual plots
        titles : List[str]
            Titles for each subplot
        layout : tuple, optional
            Grid layout (rows, cols). If None, automatically determined.

        Returns
        -------
        plt.Figure
            Combined figure
        """
        n_plots = len(plots)
        if layout is None:
            # Determine optimal layout
            cols = int(np.ceil(np.sqrt(n_plots)))
            rows = int(np.ceil(n_plots / cols))
        else:
            rows, cols = layout

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        if rows * cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i, (plot, title) in enumerate(zip(plots, titles)):
            if i < len(axes):
                # Copy plot content to subplot
                for ax in plot.axes:
                    for child in ax.get_children():
                        try:
                            axes[i].add_child(child)
                        except:
                            pass
                axes[i].set_title(title, fontsize=self.settings.font_size + 1)

        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        return fig
