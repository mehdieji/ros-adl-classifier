import sys
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm
import re
from collections import defaultdict

# --- 1. Read config ---
config_path = Path("config/plot_config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
patient_id = config["patient_id"]

# --- 2. Project paths and utils import ---
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))
from utils import get_data_path
raw_data_dir = get_data_path("raw")

# --- 3. Import your parsers ---
from data_preprocessing.m5_parser import parse_m5_sensor_data
from data_preprocessing.polar_parser import parse_polar_sensor_data
from data_preprocessing.sensomative_parser import parse_sensomative_sensor_data

# --- 4. Find all event folders for the patient ---
def extract_patient_id(folder_name):
    match = re.search(r'patient(\d+)', folder_name)
    return match.group(1) if match else None

def get_patient_folders(patient_id):
    if not raw_data_dir.exists():
        return []
    return sorted([folder for folder in raw_data_dir.iterdir()
                  if folder.is_dir() and extract_patient_id(folder.name) == patient_id])

patient_folders = get_patient_folders(patient_id)

# --- 5. Group folders by ADL event type ---
adl_events = defaultdict(list)
for folder in patient_folders:
    # Extract event type from folder name (after the last underscore)
    event_type = folder.name.split("_", maxsplit=3)[-1]
    adl_events[event_type].append(folder)

# --- 6. For each event, find all csv_dirs (instances) ---
def get_csv_dirs(event_folders):
    csv_dirs = []
    for folder in event_folders:
        subdirs = [d for d in folder.iterdir() if d.is_dir()]
        if len(subdirs) == 1:
            csv_dir = subdirs[0]
            csv_dirs.append(str(csv_dir) + "/")
    return csv_dirs

# --- 7. PDF output path ---
results_dir = project_root / "results"
results_dir.mkdir(exist_ok=True)
pdf_path = results_dir / f"patient_{patient_id}_adl_events.pdf"

# --- 8. Plotting function (from your notebook, slightly adapted) ---
def plot_event(csv_dir, event_label, pdf):
    # Parse all sensors
    m5_wrist_l = parse_m5_sensor_data(csv_dir + "M5StickC_01_data.csv")
    m5_wrist_r = parse_m5_sensor_data(csv_dir + "M5StickC_02_data.csv")
    m5_wheel   = parse_m5_sensor_data(csv_dir + "M5StickC_03_data.csv")
    polar_chest = parse_polar_sensor_data(csv_dir + "polar_acc.csv")
    sensomative_bottom = parse_sensomative_sensor_data(csv_dir + "pressure1.csv")

    sensor_data = [
        ("M5 Wrist L", m5_wrist_l, "linear_acceleration", ["linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z"], {"x": "tab:blue", "y": "tab:orange", "z": "tab:green"}),
        ("M5 Wrist R", m5_wrist_r, "linear_acceleration", ["linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z"], {"x": "tab:blue", "y": "tab:orange", "z": "tab:green"}),
        ("M5 Wheel", m5_wheel, "linear_acceleration", ["linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z"], {"x": "tab:blue", "y": "tab:orange", "z": "tab:green"}),
        ("Polar Chest", polar_chest, "linear_acceleration", ["linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z"], {"x": "tab:blue", "y": "tab:orange", "z": "tab:green"}),
        ("Sensomative Bottom", sensomative_bottom, "pressure", [f"pressure_{i}" for i in range(12)], None),
    ]

    n_sensors = len(sensor_data)
    fig, axes = plt.subplots(n_sensors, 1, figsize=(14, 2.2 * n_sensors), sharex=True)
    if n_sensors == 1:
        axes = [axes]

    for ax, (sensor_name, parsed, modality_key, cols, color_map) in zip(axes, sensor_data):
        if modality_key in parsed:
            df = parsed[modality_key]
            if sensor_name != "Sensomative Bottom":
                for axis, col in zip(["x", "y", "z"], cols):
                    if col in df.columns:
                        ax.plot(df["datetime"], df[col], label=axis, color=color_map[axis], linewidth=1)
                ax.legend(title="Axis", loc="upper right", fontsize="small")
            else:
                colors = plt.get_cmap('tab20', 12)
                for i, col in enumerate(cols):
                    if col in df.columns:
                        ax.plot(df["datetime"], df[col], label=f"cell_{i}", color=colors(i), linewidth=1)
                ax.legend(title="Cell", loc="upper right", fontsize="x-small", ncol=4)
            ax.set_ylabel(sensor_name)
        else:
            ax.set_ylabel(sensor_name)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.grid(True, linestyle=":", alpha=0.5)

    axes[-1].xaxis.set_major_locator(mdates.SecondLocator(interval=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    for label in axes[-1].get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment('right')
        label.set_fontsize(8)
    axes[-1].set_xlabel("Time")
    plt.suptitle(f"Patient {patient_id} - {event_label}", y=1.02)
    plt.tight_layout(h_pad=0.2)
    pdf.savefig(fig)
    plt.close(fig)

# --- 9. Generate the PDF ---
with PdfPages(pdf_path) as pdf:
    for adl_event, event_folders in adl_events.items():
        csv_dirs = get_csv_dirs(event_folders)
        for idx, csv_dir in enumerate(csv_dirs):
            event_label = f"{adl_event} (instance {idx+1})"
            print(f"Plotting: {event_label} from {csv_dir}")
            try:
                plot_event(csv_dir, event_label, pdf)
            except Exception as e:
                print(f"Failed to plot {event_label}: {e}")

print(f"PDF saved to: {pdf_path}") 