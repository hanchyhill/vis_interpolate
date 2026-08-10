# -*- coding: utf-8 -*-
r"""
Visibility Model Comparison Statistics and Visualization

Task: Generate comparison statistics plots and save statistics to CSV files.
Compare three visibility models: CLDAS, DEM_N, DEM_NR

Data files:
-------------------------------
CLDAS model:
- H:\github\python\vis_interpolate\data\cldas_score\vis_score_summary_national.csv
- H:\github\python\vis_interpolate\data\cldas_score\vis_score_summary_regional.csv
-------------------------------
DEM_N model:
- H:\github\python\vis_interpolate\data\model_score\national\vis_score_summary_national_model_national.csv
- H:\github\python\vis_interpolate\data\model_score\national\vis_score_summary_regional_model_national.csv
-------------------------------
DEM_NR model:
- H:\github\python\vis_interpolate\data\model_score\national_and_regional\vis_score_summary_national_model_national_and_regional.csv
- H:\github\python\vis_interpolate\data\model_score\national_and_regional\vis_score_summary_regional_model_national_and_regional.csv
-------------------------------

CSV fields used: std_error, bias, correlation, datetime

Output:
1. Boxplots showing overall std_error, correlation, bias for each model
2. Monthly line plots showing average metrics over time
3. Separate plots for national and regional stations (4 plots + 4 CSV files total)
4. All outputs saved to: H:\github\python\vis_interpolate\data\plot\statistic
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# Chinese font settings
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Data file paths configuration
DATA_PATHS = {
    'CLDAS': {
        'national': r"H:\github\python\vis_interpolate\data\cldas_score\vis_score_summary_national.csv",
        'regional': r"H:\github\python\vis_interpolate\data\cldas_score\vis_score_summary_regional.csv"
    },
    'DEM_N': {
        'national': r"H:\github\python\vis_interpolate\data\model_score\national\vis_score_summary_national_model_national.csv",
        'regional': r"H:\github\python\vis_interpolate\data\model_score\national\vis_score_summary_regional_model_national.csv"
    },
    'DEM_NR': {
        'national': r"H:\github\python\vis_interpolate\data\model_score\national_and_regional\vis_score_summary_national_model_national_and_regional.csv",
        'regional': r"H:\github\python\vis_interpolate\data\model_score\national_and_regional\vis_score_summary_regional_model_national_and_regional.csv"
    }
}

# Output directory
OUTPUT_DIR = Path(r"H:\github\python\vis_interpolate\data\plot\statistic")


def load_model_data(station_type: str) -> Dict[str, pd.DataFrame]:
    """
    Load all model data for specified station type

    Args:
        station_type: 'national' or 'regional'

    Returns:
        Dictionary with model names as keys and DataFrames as values
    """
    data = {}
    for model_name, paths in DATA_PATHS.items():
        try:
            df = pd.read_csv(paths[station_type])
            df['datetime'] = pd.to_datetime(df['datetime'])
            data[model_name] = df
            print(f"Loaded {model_name} ({station_type}): {len(df)} records")
        except Exception as e:
            print(f"Warning: Cannot load {model_name} ({station_type}): {e}")

    return data


def create_boxplot(data: Dict[str, pd.DataFrame], station_type: str, output_dir: Path):
    """
    Create boxplot and save statistics

    Args:
        data: Dictionary of model data
        station_type: 'national' or 'regional'
        output_dir: Output directory
    """
    metrics = ['std_error', 'bias', 'correlation']
    metric_names = {'std_error': 'Std Error', 'bias': 'Bias', 'correlation': 'Correlation'}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    station_label = 'National Stations' if station_type == 'national' else 'Regional Stations'
    fig.suptitle(f'Model Performance Comparison - {station_label}',
                 fontsize=16, fontweight='bold')

    # Store statistics
    stats_data = []

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Prepare boxplot data
        box_data = []
        labels = []

        for model_name in ['CLDAS', 'DEM_N', 'DEM_NR']:
            if model_name in data:
                values = data[model_name][metric].dropna()
                box_data.append(values)
                labels.append(model_name)

                # Calculate statistics
                stats = {
                    'Model': model_name,
                    'Metric': metric_names[metric],
                    'Mean': values.mean(),
                    'Median': values.median(),
                    'Std': values.std(),
                    'Min': values.min(),
                    'Max': values.max(),
                    'Q25': values.quantile(0.25),
                    'Q75': values.quantile(0.75),
                    'Q90': values.quantile(0.90),
                    'Q95': values.quantile(0.95)
                }
                stats_data.append(stats)

        # Draw boxplot
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True,
                        showmeans=True, meanline=True)

        # Set colors
        colors = ['#FF6B6B', '#2ECC71', '#3498DB']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel(metric_names[metric], fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_title(f'{metric_names[metric]} Distribution', fontsize=14)

        # Highlight zero line
        ax.axhline(y=0, color='red', linewidth=2, linestyle='-', alpha=0.8, zorder=5)

    plt.tight_layout()

    # Save figure
    fig_path = output_dir / f"boxplot_{station_type}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Boxplot saved: {fig_path}")
    plt.close()

    # Save statistics
    stats_df = pd.DataFrame(stats_data)
    csv_path = output_dir / f"boxplot_stats_{station_type}.csv"
    stats_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Boxplot statistics saved: {csv_path}")


def create_monthly_lineplot(data: Dict[str, pd.DataFrame], station_type: str, output_dir: Path):
    """
    Create monthly line plot and save statistics

    Args:
        data: Dictionary of model data
        station_type: 'national' or 'regional'
        output_dir: Output directory
    """
    metrics = ['std_error', 'bias', 'correlation']
    metric_names = {'std_error': 'Std Error', 'bias': 'Bias', 'correlation': 'Correlation'}

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    station_label = 'National Stations' if station_type == 'national' else 'Regional Stations'
    fig.suptitle(f'Monthly Average Performance Comparison - {station_label}',
                 fontsize=16, fontweight='bold')

    # Store statistics
    monthly_stats = []

    colors = {'CLDAS': '#FF6B6B', 'DEM_N': '#2ECC71', 'DEM_NR': '#3498DB'}
    markers = {'CLDAS': 'o', 'DEM_N': 's', 'DEM_NR': '^'}

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        for model_name in ['CLDAS', 'DEM_N', 'DEM_NR']:
            if model_name in data:
                df = data[model_name].copy()
                df['year_month'] = df['datetime'].dt.to_period('M')

                # Calculate monthly average
                monthly_avg = df.groupby('year_month')[metric].mean()

                # Convert to continuous month index
                months = [f"{ym}" for ym in monthly_avg.index]
                values = monthly_avg.values

                # Save statistics
                for month, value in zip(months, values):
                    monthly_stats.append({
                        'YearMonth': str(month),
                        'Model': model_name,
                        'Metric': metric_names[metric],
                        'Average': value
                    })

                # Plot line
                ax.plot(range(len(months)), values,
                       color=colors[model_name],
                       marker=markers[model_name],
                       label=model_name,
                       linewidth=2,
                       markersize=6,
                       alpha=0.8)

        ax.set_ylabel(metric_names[metric], fontsize=12)
        ax.set_xlabel('Month', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_title(f'Monthly Average {metric_names[metric]}', fontsize=14)

        # Highlight zero line
        ax.axhline(y=0, color='red', linewidth=2, linestyle='-', alpha=0.8, zorder=5)

        # Set x-axis labels (only show some months to avoid crowding)
        if len(months) > 0:
            step = max(1, len(months) // 12)
            ax.set_xticks(range(0, len(months), step))
            ax.set_xticklabels([months[i] for i in range(0, len(months), step)],
                              rotation=45, ha='right')

    plt.tight_layout()

    # Save figure
    fig_path = output_dir / f"monthly_lineplot_{station_type}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Monthly lineplot saved: {fig_path}")
    plt.close()

    # Save statistics
    monthly_df = pd.DataFrame(monthly_stats)
    csv_path = output_dir / f"monthly_stats_{station_type}.csv"
    monthly_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Monthly statistics saved: {csv_path}")


def main():
    """Main function"""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Visibility Model Comparison Statistics")
    print("="*60)

    for station_type in ['national', 'regional']:
        station_name = "National" if station_type == "national" else "Regional"
        print(f"\nProcessing {station_name} data...")
        print("-"*60)

        # Load data
        data = load_model_data(station_type)

        if not data:
            print(f"Warning: No available {station_name} data")
            continue

        # Create boxplot
        print(f"\nGenerating {station_name} boxplot...")
        create_boxplot(data, station_type, OUTPUT_DIR)

        # Create monthly lineplot
        print(f"\nGenerating {station_name} monthly lineplot...")
        create_monthly_lineplot(data, station_type, OUTPUT_DIR)

    print("\n" + "="*60)
    print("All charts and statistics generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
