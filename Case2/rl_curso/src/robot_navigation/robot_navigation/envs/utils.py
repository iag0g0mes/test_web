import math
import numpy as np
from typing import Tuple
from dataclasses import fields
from rich.console import Console
from rich.table import Table

def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

def quat_to_yaw(x: float, 
                y: float, 
                z: float, 
                w: float
) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

def yaw_to_quat(
    yaw:float
)->Tuple[float, float, float, float]:
    
    half = yaw / 2.0
    w = math.cos(half)
    z = math.sin(half)
    
    # x, y, z, w
    return 0.0, 0.0, float(z), float(w)

def downsample_scan(ranges: np.ndarray, n_beams: int) -> np.ndarray:
    if ranges.ndim != 1:
        ranges = ranges.flatten()
    if len(ranges) == n_beams:
        return ranges
    idx = np.linspace(0, len(ranges) - 1, n_beams).astype(np.int32)
    return ranges[idx]

def sanitize_scan(ranges: np.ndarray, range_min: float, range_max: float) -> np.ndarray:
    r = np.array(ranges, dtype=np.float32)
    r = np.nan_to_num(r, nan=range_max, posinf=range_max, neginf=range_min)
    r = np.clip(r, range_min, range_max)
    return r

def tuple_starts(x) -> Tuple[Tuple[float, float, float], ...]:
    return tuple((float(a), float(b), float(c)) for a, b, c in x)

def tuple_goals(x) -> Tuple[Tuple[float, float], ...]:
    return tuple((float(a), float(b)) for a, b in x)

def print_config_rich(obj):
    console = Console()
    
    table = Table(title=obj.__class__.__name__, title_justify="left",)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for f in fields(obj):
        value = getattr(obj, f.name)
        if f.name in {"starts", "goals"}:
            value = f"{len(value)} points"
        table.add_row(f.name, str(value))

    print("")
    console.print(table)
    print("")