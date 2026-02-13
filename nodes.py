import torch
import safetensors.torch
import json
import os
import re
import sys
import warnings
import folder_paths
import comfy.utils
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional, Union, Dict, Tuple, List, Set

# Suppress specific Pynvml warning
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")

# ============================================================================
# OPTIONAL IMPORTS & GLOBALS
# ============================================================================

try:
    from comfy_kitchen.tensor import QuantizedTensor
    import comfy_kitchen as ck

    COMFY_KITCHEN_AVAILABLE = True
except ImportError:
    COMFY_KITCHEN_AVAILABLE = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False

# Configuration Files (Saved in the node's directory)
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
PRESET_FILENAME = os.path.join(BASE_DIR, "quantization_presets.json")
MODEL_TYPES_FILENAME = os.path.join(BASE_DIR, "model_types.json")
CALIBRATION_FILENAME = os.path.join(BASE_DIR, "calibration_scales.json")

# ============================================================================
# NVFP4 SCALE CONSTANTS
# ============================================================================
# These must match comfy-kitchen's convention:
#   scale = amax / (F8_E4M3_MAX * F4_E2M1_MAX)
# The scaled_mm_nvfp4 kernel uses tensor_scale_a * tensor_scale_b * (block_A @ block_B)
# so both weight_scale_2 and input_scale must be normalized by this factor.
F8_E4M3_MAX = 448.0
F4_E2M1_MAX = 6.0
NVFP4_SCALE_DIVISOR = F8_E4M3_MAX * F4_E2M1_MAX  # 2688.0

# ============================================================================
# CORE UTILITIES (Ported from Script)
# ============================================================================


def natural_sort_key(s: str) -> List:
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


def normalize_layer_name(name: str) -> str:
    patterns = [r"^model\.diffusion_model\.", r"^diffusion_model\.", r"^model\."]
    normalized = name
    for pattern in patterns:
        normalized = re.sub(pattern, "", normalized)
    return normalized


def load_json(filename: str) -> Dict:
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def save_json(data: Dict, filename: str):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def get_quant_label(
    weights: Dict[str, torch.Tensor], key: str, original_key: str
) -> str:
    if original_key not in weights:
        return "-"
    tensor = weights[original_key]
    dtype_str = str(tensor.dtype).replace("torch.", "")

    if dtype_str == "uint8":
        candidates = [
            original_key + ".weight_scale_2",
            original_key + "_scale_2",
            original_key.replace(".weight", ".weight_scale_2"),
        ]
        if any(c in weights for c in candidates):
            return "nvfp4"

    if "float8_e4m3" in dtype_str:
        return "fp8_e4m3fn"
    if "float8_e5m2" in dtype_str:
        return "fp8_e5m2"
    if "bfloat16" in dtype_str:
        return "bf16"
    return dtype_str


def decompose_lora_from_diff(
    diff: torch.Tensor, name: str, rank: int
) -> Dict[str, torch.Tensor]:
    if not name.endswith(".weight"):
        return {}
    if rank < 1:
        raise ValueError("LoRA rank must be >= 1")

    # Ensure float32 for SVD stability and use CUDA if available for speed
    # SVD on CPU is extremely slow for large matrices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diff_f32 = diff.to(device=device, dtype=torch.float32)

    U, S, Vh = torch.linalg.svd(diff_f32, full_matrices=False)

    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]

    sqrt_S_r = torch.diag(torch.sqrt(S_r))
    lora_B = U_r @ sqrt_S_r
    lora_A = sqrt_S_r @ Vh_r

    # Convert back to original dtype if needed, or keep float
    base_name = name.removesuffix(".weight")
    return {
        f"{base_name}.lora_A.weight": lora_A.to(
            diff.dtype
        ).cpu(),  # Move back to CPU to save GPU RAM
        f"{base_name}.lora_B.weight": lora_B.to(diff.dtype).cpu(),
        f"{base_name}.alpha": torch.tensor(float(rank), dtype=torch.float32),
    }


# ============================================================================
# QUANTIZATION LOGIC
# ============================================================================


def quantize_nvfp4_weights(
    tensor: torch.Tensor,
    name: str,
    lora_dict: Dict,
    svd_rank: Optional[int],
    input_scale_val: Optional[float] = None,
) -> Tuple[Dict, Dict, Dict]:
    if not COMFY_KITCHEN_AVAILABLE:
        raise RuntimeError("comfy-kitchen required for nvfp4")

    # Cast to half for quantization
    if tensor.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        tensor = tensor.half()

    qt = QuantizedTensor.from_float(tensor, "TensorCoreNVFP4Layout")

    # SVD Correction
    if svd_rank is not None:
        # Optimization: Don't move to CPU if not needed, but qt.dequantize() might return on device
        # diff calculation should happen on same device
        diff = tensor - qt.dequantize()
        # Merge new lora weights into the dictionary (Python 3.9+ | operator)
        lora_dict = lora_dict | decompose_lora_from_diff(diff, name, svd_rank)

    q_data = {"format": "nvfp4"}

    out_tensors = {
        f"{name}": qt._qdata,
        f"{name}_scale": qt.params.block_scale,
        f"{name}_scale_2": qt.params.scale,
    }

    # Inject Input Scale if calibrated
    # Do NOT add comfy_quant tensor - it's not needed and causes compatibility issues
    if input_scale_val is not None:
        out_tensors[f"{name.removesuffix('weight')}input_scale"] = torch.tensor(
            input_scale_val, dtype=torch.float32
        )
        q_data["input_scale"] = input_scale_val

    return out_tensors, {name.removesuffix(".weight"): q_data}, lora_dict


def quantize_fp8_scaled_weights(
    tensor: torch.Tensor, name: str, lora_dict: Dict, svd_rank: Optional[int]
) -> Tuple[Dict, Dict, Dict]:
    if not COMFY_KITCHEN_AVAILABLE:
        raise RuntimeError("comfy-kitchen required for fp8_scaled")

    qt = QuantizedTensor.from_float(tensor, "TensorCoreFP8Layout")

    if svd_rank is not None:
        diff = tensor - qt.dequantize()
        lora_dict = lora_dict | decompose_lora_from_diff(diff, name, svd_rank)

    q_data = {"format": "float8_e4m3fn"}
    return (
        {
            f"{name}": qt._qdata,
            f"{name}_scale": qt.params.scale,
        },
        {name.removesuffix(".weight"): q_data},
        lora_dict,
    )


def name_to_format(name: str):
    if not isinstance(name, str):
        return None
    normalized = name.lower().replace("float", "fp")
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp8_e4m3fn": torch.float8_e4m3fn,
        "fp8": torch.float8_e4m3fn,
    }
    if normalized in ["nvfp4", "fp4"]:
        return "nvfp4"
    if normalized in ["fp8_scaled", "scaled_fp8"]:
        return "fp8_scaled"
    return dtype_map.get(normalized, None)


def to_format(tensor, name, format_spec, lora_dict, svd_rank, input_scale=None):
    if isinstance(format_spec, torch.dtype):
        converted = tensor.to(format_spec)
        if svd_rank is not None:
            # Cast back to original for diff
            diff = tensor - converted.to(tensor.dtype)
            lora_dict = lora_dict | decompose_lora_from_diff(diff, name, svd_rank)
        return {name: converted}, {}, lora_dict

    format_lower = format_spec.lower().replace("float", "fp")
    if format_lower in ["nvfp4", "fp4"]:
        return quantize_nvfp4_weights(tensor, name, lora_dict, svd_rank, input_scale)
    elif format_lower in ["fp8_scaled", "scaled_fp8"]:
        return quantize_fp8_scaled_weights(tensor, name, lora_dict, svd_rank)
    else:
        fmt = name_to_format(format_spec)
        if fmt is None:
            raise ValueError(f"Unknown format: {format_spec}")
        return to_format(tensor, name, fmt, lora_dict, svd_rank)


# ============================================================================
# REPORT GENERATION HELPER
# ============================================================================


def generate_text_table(headers, rows):
    """Simple text table generator for UI display where Rich isn't available."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    fmt = " | ".join([f"{{:<{w}}}" for w in col_widths])
    lines = [fmt.format(*headers), "-" * (sum(col_widths) + 3 * (len(headers) - 1))]
    for row in rows:
        lines.append(fmt.format(*[str(c) for c in row]))
    return "\n".join(lines)


# ============================================================================
# NODE 1: INSPECT MODEL
# ============================================================================


class DiTInspectNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_filename": (folder_paths.get_filename_list("diffusion_models"),),
                "register_as": ("STRING", {"default": "", "multiline": False}),
                "set_model_type": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "Save/Register",
                        "label_off": "Just Inspect",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report_text",)
    FUNCTION = "inspect"
    CATEGORY = "Quantization/Ops"

    def inspect(self, model_filename, register_as, set_model_type):
        model_path = folder_paths.get_full_path("diffusion_models", model_filename)
        if not model_path:
            return (f"Error: Model {model_filename} not found.",)

        # Logic from inspect_layers
        weights = safetensors.torch.load_file(model_path)

        # Identify Type
        model_types = load_json(MODEL_TYPES_FILENAME)
        try:
            current_type = identify_model_type_internal(weights, model_types)
        except:
            current_type = "Unknown"

        # Registration Logic
        msg = ""
        if set_model_type and register_as.strip():
            ignored_suffixes = [
                ".weight_scale",
                ".weight_scale_2",
                ".input_scale",
                ".scale_weight",
                ".comfy_quant",
            ]
            normalized_keys = []
            for k in weights.keys():
                if any(k.endswith(s) for s in ignored_suffixes):
                    continue
                normalized_keys.append(normalize_layer_name(k))
            normalized_keys.sort(key=natural_sort_key)

            model_types[register_as] = normalized_keys
            save_json(model_types, MODEL_TYPES_FILENAME)
            msg = f"\n✅ Registered model type '{register_as}' to {MODEL_TYPES_FILENAME}\n"
            print(f"[ComfyUI-NVFP4-Quantizer] Registered type: {register_as}")
            current_type = register_as

        # Table Generation
        rows = []
        total_size = 0
        sorted_keys = sorted(weights.keys(), key=natural_sort_key)

        for i, key in enumerate(sorted_keys, 1):
            tensor = weights[key]
            shape = "x".join(map(str, tensor.shape)) if tensor.dim() > 0 else "1"
            quant_fmt = get_quant_label(weights, key, key)
            size_mb = tensor.nelement() * tensor.element_size() / (1024 * 1024)
            total_size += size_mb
            rows.append(
                [str(i), normalize_layer_name(key), shape, quant_fmt, f"{size_mb:.2f}"]
            )

        # Console Output (Rich)
        if RICH_AVAILABLE:
            table = Table(title=f"Inspect: {model_filename} ({current_type})")
            table.add_column("#")
            table.add_column("Layer")
            table.add_column("Shape")
            table.add_column("Quant")
            table.add_column("MB")
            for r in rows:
                table.add_row(*r)
            console.print(table)
            console.print(msg)

        # UI Output (Text)
        text_report = (
            f"Model: {model_filename}\nType: {current_type}\nTotal Size: {total_size:.2f} MB\n"
            + msg
            + "\n"
        )
        text_report += generate_text_table(
            ["#", "Layer Name", "Shape", "Quant", "Size(MB)"], rows
        )

        return (text_report,)


def identify_model_type_internal(weights, model_types_data):
    ignored = [
        ".weight_scale",
        ".weight_scale_2",
        ".input_scale",
        ".scale_weight",
        ".comfy_quant",
    ]
    current_keys = set()
    for k in weights.keys():
        if any(k.endswith(s) for s in ignored):
            continue
        current_keys.add(normalize_layer_name(k))

    for m_name, m_keys in model_types_data.items():
        if set(m_keys) == current_keys:
            return m_name
    raise ValueError("Unknown")


# ============================================================================
# NODE 2: COMPARE MODELS
# ============================================================================


class DiTCompareNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_a": (folder_paths.get_filename_list("diffusion_models"),),
                "model_b": (folder_paths.get_filename_list("diffusion_models"),),
                "infer_preset_name": ("STRING", {"default": "", "multiline": False}),
                "infer_quant_preset": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "Save Preset",
                        "label_off": "Just Compare",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report_text",)
    FUNCTION = "compare"
    CATEGORY = "Quantization/Ops"

    def compare(self, model_a, model_b, infer_preset_name, infer_quant_preset):
        path_a = folder_paths.get_full_path("diffusion_models", model_a)
        path_b = folder_paths.get_full_path("diffusion_models", model_b)

        w1 = safetensors.torch.load_file(path_a)
        w2 = safetensors.torch.load_file(path_b)

        ignored = [
            ".weight_scale",
            ".weight_scale_2",
            ".input_scale",
            ".scale_weight",
            ".comfy_quant",
        ]
        global_map = defaultdict(dict)

        # Map logical layers
        for idx, w in enumerate([w1, w2]):
            for k in w.keys():
                if any(k.endswith(s) for s in ignored):
                    continue
                norm = normalize_layer_name(k)
                global_map[norm][idx] = k

        all_keys = sorted(global_map.keys(), key=natural_sort_key)
        rows = []
        changes = {}

        for i, norm_key in enumerate(all_keys, 1):
            k1 = global_map[norm_key].get(0)
            k2 = global_map[norm_key].get(1)

            q1 = get_quant_label(w1, norm_key, k1) if k1 else "Missing"
            q2 = get_quant_label(w2, norm_key, k2) if k2 else "Missing"

            status = "Same"
            if q1 != q2:
                status = "Diff"
                if k1 and k2:
                    changes[norm_key] = q2

            # Get size of M2 for reference
            size = 0
            if k2:
                t = w2[k2]
                size = t.nelement() * t.element_size() / (1024 * 1024)

            rows.append([str(i), norm_key, q1, q2, f"{size:.2f}", status])

        msg = ""
        # Inference Logic
        if infer_quant_preset and infer_preset_name.strip():
            model_types = load_json(MODEL_TYPES_FILENAME)
            try:
                m_type = identify_model_type_internal(w1, model_types)
                presets = load_json(PRESET_FILENAME)
                if m_type not in presets:
                    presets[m_type] = {}
                presets[m_type][infer_preset_name] = changes
                save_json(presets, PRESET_FILENAME)
                msg = f"\n✅ Saved preset '{infer_preset_name}' for type '{m_type}' with {len(changes)} rules.\n"
                print(f"[ComfyUI-NVFP4-Quantizer] Saved preset: {infer_preset_name}")
            except Exception as e:
                msg = f"\n❌ Could not infer preset: {str(e)}\n"
                print(f"[ComfyUI-NVFP4-Quantizer] Error inferring preset: {e}")

        # Console Output
        if RICH_AVAILABLE:
            table = Table(title=f"Compare: {model_a} vs {model_b}")
            table.add_column("#")
            table.add_column("Layer")
            table.add_column("M1 Q")
            table.add_column("M2 Q")
            table.add_column("M2 MB")
            table.add_column("Status")
            for r in rows:
                table.add_row(*r)
            console.print(table)
            console.print(msg)

        text_report = f"Comparison: {model_a} vs {model_b}\n" + msg + "\n"
        text_report += generate_text_table(
            ["#", "Layer", "M1 Q", "M2 Q", "M2 MB", "Status"], rows
        )
        return (text_report,)


# ============================================================================
# NODE 3: CHECK EQUALITY
# ============================================================================


class DiTIsEqualNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_a": (folder_paths.get_filename_list("diffusion_models"),),
                "model_b": (folder_paths.get_filename_list("diffusion_models"),),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report_text",)
    FUNCTION = "check_equality"
    CATEGORY = "Quantization/Ops"

    def check_equality(self, model_a, model_b):
        path_a = folder_paths.get_full_path("diffusion_models", model_a)
        path_b = folder_paths.get_full_path("diffusion_models", model_b)

        print(f"[ComfyUI-NVFP4-Quantizer] Checking equality: {model_a} vs {model_b}")

        w1 = safetensors.torch.load_file(path_a)
        w2 = safetensors.torch.load_file(path_b)

        all_keys = sorted(set(w1.keys()) | set(w2.keys()), key=natural_sort_key)
        rows = []
        equal_cnt = 0
        issue_cnt = 0

        for i, key in enumerate(all_keys, 1):
            status = "-"
            shape_str = "-"
            m1_dtype = "-"
            m2_dtype = "-"
            diff_max_str = "-"
            diff_avg_str = "-"
            larger_str = "-"
            ratio_str = "-"
            m1_val_str = "-"
            m2_val_str = "-"

            # Extract dtypes
            if key in w1:
                m1_dtype = str(w1[key].dtype).replace("torch.", "")
            if key in w2:
                m2_dtype = str(w2[key].dtype).replace("torch.", "")

            if key not in w1:
                status = "Missing M1"
                issue_cnt += 1
            elif key not in w2:
                status = "Missing M2"
                issue_cnt += 1
            else:
                t1, t2 = w1[key], w2[key]
                # Format Shape (show "1" for scalars)
                if t1.dim() == 0:
                    s_str = "1"
                else:
                    s_str = str(list(t1.shape))

                if t1.dim() == 0 and t2.dim() == 0:
                    m1_val_str = f"{t1.float().item():.6e}"
                    m2_val_str = f"{t2.float().item():.6e}"

                if t1.shape != t2.shape:
                    status = "Shape Diff"
                    shape_str = f"{s_str} vs {list(t2.shape)}"
                    issue_cnt += 1
                else:
                    shape_str = s_str
                    # Compute Differences and Norms
                    t1f, t2f = t1.float(), t2.float()
                    abs_diff = (t1f - t2f).abs()

                    diff_max = abs_diff.max().item()
                    diff_avg = abs_diff.mean().item()

                    diff_max_str = f"{diff_max:.2e}"
                    diff_avg_str = f"{diff_avg:.2e}"

                    # Compute L2 Norms for Magnitude Comparison
                    norm1 = torch.norm(t1f).item()
                    norm2 = torch.norm(t2f).item()

                    # Who is larger?
                    if norm1 > norm2:
                        larger_str = "M1"
                    elif norm2 > norm1:
                        larger_str = "M2"
                    else:
                        larger_str = "="

                    # Ratio (L2 / L2)
                    if norm2 != 0:
                        ratio = norm1 / norm2
                        ratio_str = f"{ratio:.4f}"
                    else:
                        ratio_str = "inf"

                    if t1.dtype == torch.uint8:
                        if torch.equal(t1, t2):
                            status = "Equal"
                            diff_max_str = "0.0"
                            equal_cnt += 1
                        else:
                            status = "Not Equal"
                            issue_cnt += 1
                    else:
                        # Float tolerance check
                        if diff_max < 1e-5:
                            status = "Equal"
                            equal_cnt += 1
                        else:
                            status = "Not Equal"
                            issue_cnt += 1

            rows.append(
                [
                    str(i),
                    key,
                    shape_str,
                    m1_dtype,
                    m2_dtype,
                    status,
                    diff_max_str,
                    diff_avg_str,
                    larger_str,
                    ratio_str,
                    m1_val_str,
                    m2_val_str,
                ]
            )

        if RICH_AVAILABLE:
            table = Table(title=f"IsEqual: {model_a} vs {model_b}")
            table.add_column("#")
            table.add_column("Layer")
            table.add_column("Shape")
            table.add_column("M1 Type")
            table.add_column("M2 Type")
            table.add_column("Status")
            table.add_column("Max Diff")
            table.add_column("Avg Diff")
            table.add_column("Larger?")
            table.add_column("Ratio")
            # Note: rows contains 12 items, but we only defined 10 columns here for Rich + title columns.
            # To avoid mismatch errors in Rich if the row length exceeds columns, we slice the row.
            # We skip the last two items (m1_val_str, m2_val_str) for the console table to keep it compact.
            for r in rows:
                table.add_row(*r[:10])
            console.print(table)
            console.print(f"Equal: {equal_cnt}, Issues: {issue_cnt}")

        text_report = f"Equality Check\nM1: {model_a}\nM2: {model_b}\nSummary: Equal: {equal_cnt}, Issues: {issue_cnt}\n\n"
        text_report += generate_text_table(
            [
                "#",
                "Layer",
                "Shape",
                "M1 Type",
                "M2 Type",
                "Status",
                "MaxDiff",
                "AvgDiff",
                "Larger?",
                "Ratio",
                "M1 Val",
                "M2 Val",
            ],
            rows,
        )
        return (text_report,)


# ============================================================================
# NODE 4: QUANTIZE MODEL
# ============================================================================


class DiTQuantizeNode:
    @classmethod
    def INPUT_TYPES(s):
        # Always reload JSONs to support "Force Refresh" behavior on client side logic (if cached there)
        # and ensure server side always has latest data
        m_types = list(load_json(MODEL_TYPES_FILENAME).keys())

        presets = []
        all_presets = load_json(PRESET_FILENAME)
        for t in all_presets:
            for p in all_presets[t]:
                presets.append(f"{t}::{p}")  # Namespaced presets

        # Load Calibration Scale Profiles from internal file
        # Format: Type::Profile
        calib_options = ["None"]
        calib_data = load_json(CALIBRATION_FILENAME)
        for m_type, profiles in calib_data.items():
            for profile in profiles:
                calib_options.append(f"{m_type}::{profile}")

        return {
            "required": {
                "model_filename": (folder_paths.get_filename_list("diffusion_models"),),
                "model_type": (m_types if m_types else ["None"],),
                "preset_combo": (presets if presets else ["None"],),
                "calibration_combo": (calib_options,),
                "compute_residual_lora": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "Compute & Save LoRA",
                        "label_off": "No LoRA",
                    },
                ),
                "svd_rank": ("INT", {"default": 0, "min": 0, "max": 128}),
                "calibration_margin": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.05,
                        "tooltip": "Multiplier to increase calibrated input scales (1 + margin). Helps prevent clipping outliers.",
                    },
                ),
                "cap_input_scale": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "Cap Input Scale",
                        "label_off": "No Cap",
                    },
                ),
                "input_scale_cap": (
                    "FLOAT",
                    {
                        "default": 10.0,
                        "min": 0.1,
                        "max": 10000.0,
                        "step": 0.1,
                        "tooltip": "Maximum allowed value for input_scale. Prevents signal crushing in high-variance layers.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status_text",)
    FUNCTION = "quantize"
    CATEGORY = "Quantization/Core"

    def quantize(
        self,
        model_filename,
        model_type,
        preset_combo,
        calibration_combo,
        compute_residual_lora,
        svd_rank,
        calibration_margin,
        cap_input_scale,
        input_scale_cap,
    ):
        if "::" not in preset_combo:
            return ("Error: Invalid preset selection",)
        _, preset_name = preset_combo.split("::", 1)

        # Load Resources
        model_path = folder_paths.get_full_path("diffusion_models", model_filename)
        presets_data = load_json(PRESET_FILENAME)
        target_map = presets_data.get(model_type, {}).get(preset_name, None)

        if not target_map:
            return (f"Error: Preset {preset_name} not found for type {model_type}",)

        calib_data = {}
        calib_name = "NoCalib"
        if calibration_combo != "None" and "::" in calibration_combo:
            c_type, c_profile = calibration_combo.split("::", 1)
            all_calib = load_json(CALIBRATION_FILENAME)
            if c_type in all_calib and c_profile in all_calib[c_type]:
                calib_data = all_calib[c_type][c_profile]
                calib_name = c_profile
                print(
                    f"[ComfyUI-NVFP4-Quantizer] Loaded calibration profile: {c_profile} with {len(calib_data)} layers"
                )
            else:
                print(
                    f"[ComfyUI-NVFP4-Quantizer] Warning: Calibration {calibration_combo} not found."
                )

        print(f"[ComfyUI-NVFP4-Quantizer] Loading {model_path}...")
        weights = safetensors.torch.load_file(model_path)

        # Prepare Output
        out_model = {}
        quant_metadata = {}
        lora_model = {}

        all_keys = sorted(weights.keys(), key=natural_sort_key)

        # Progress Bar
        layers_to_convert = [
            k for k in all_keys if normalize_layer_name(k) in target_map
        ]
        pbar = comfy.utils.ProgressBar(len(layers_to_convert))

        print(
            f"[ComfyUI-NVFP4-Quantizer] Quantizing {len(layers_to_convert)} layers..."
        )

        # SVD Speedup: Only compute if enabled and rank > 0
        effective_rank = svd_rank if (compute_residual_lora and svd_rank > 0) else None

        for key in all_keys:
            norm_key = normalize_layer_name(key)
            if norm_key in target_map:
                target_fmt = target_map[norm_key]
                tensor = weights.pop(key)

                # Check calibration - match layer names correctly
                scale_val = None
                base_key = normalize_layer_name(key.removesuffix(".weight"))
                if base_key in calib_data:
                    raw_scale_val = calib_data[base_key]
                    # Apply Safety Margin (Multiplier)
                    # scale = max_val / const. Increasing scale increases the dynamic range covered.
                    scale_val = raw_scale_val * (1.0 + calibration_margin)

                    # Apply Cap logic
                    if cap_input_scale:
                        scale_val = min(scale_val, input_scale_cap)

                    print(
                        f"[ComfyUI-NVFP4-Quantizer] Using calibration for {base_key}: {raw_scale_val:.4e} -> {scale_val:.4e} (Margin: {calibration_margin}, Cap: {input_scale_cap if cap_input_scale else 'Off'})"
                    )

                # IMPORTANT: Pass empty dict for lora accumulator to keep loop clean, then update main dict
                new_w, meta, new_lora_partial = to_format(
                    tensor, key, target_fmt, {}, effective_rank, scale_val
                )

                out_model.update(new_w)
                quant_metadata.update(meta)
                # CRITICAL BUG FIX: Update the main lora_model dictionary with results from this layer
                if new_lora_partial:
                    lora_model.update(new_lora_partial)

                pbar.update(1)
            else:
                out_model[key] = weights.pop(key)  # Keep original

        # Normalize all keys to remove prefixes consistently
        normalized_out_model = {
            normalize_layer_name(k): v for k, v in out_model.items()
        }
        out_model = normalized_out_model

        normalized_meta = {
            normalize_layer_name(k): v for k, v in quant_metadata.items()
        }
        quant_metadata = normalized_meta

        if lora_model:
            normalized_lora = {
                normalize_layer_name(k): v for k, v in lora_model.items()
            }
            lora_model = normalized_lora

        # Construct Filename and Path
        # Default: /models/diffusion_models/<orig>_<preset>_<calib>.safetensors
        base_name = os.path.splitext(os.path.basename(model_filename))[0]
        output_filename = f"{base_name}_{preset_name}_{calib_name}.safetensors"

        # Try to save in same folder as original if possible, else models/diffusion_models root
        original_dir = os.path.dirname(model_path)
        if os.access(original_dir, os.W_OK):
            out_path = os.path.join(original_dir, output_filename)
        else:
            # Fallback to standard ComfyUI models folder
            out_path = os.path.join(
                folder_paths.get_folder_paths("diffusion_models")[0], output_filename
            )

        metadata = (
            {
                "_quantization_metadata": json.dumps(
                    {"format_version": "1.0", "layers": quant_metadata}
                )
            }
            if quant_metadata
            else None
        )
        safetensors.torch.save_file(out_model, out_path, metadata=metadata)
        print(f"[ComfyUI-NVFP4-Quantizer] Saved model to: {out_path}")

        status_msg = f"Quantized Model Saved:\n{out_path}"

        # Save LoRA if requested and data exists
        if compute_residual_lora and lora_model:
            lora_filename = f"{base_name}_{preset_name}_{calib_name}_lora.safetensors"

            # Use ComfyUI standard loras folder
            lora_dirs = folder_paths.get_folder_paths("loras")
            if lora_dirs:
                lora_out_path = os.path.join(lora_dirs[0], lora_filename)
            else:
                # Fallback to output folder
                lora_out_path = os.path.join(os.path.dirname(out_path), lora_filename)

            safetensors.torch.save_file(lora_model, lora_out_path)
            print(f"[ComfyUI-NVFP4-Quantizer] Saved SVD LoRA: {lora_out_path}")
            status_msg += f"\n\nResidual LoRA Saved:\n{lora_out_path}"
        elif compute_residual_lora and not lora_model:
            status_msg += "\n\n(No residual LoRA generated; effective rank was 0)"

        return (status_msg,)


# ============================================================================
# CALIBRATION NODES (Robust MAX Approach)
# ============================================================================

# Global context to persist stats across multiple ComfyUI prompt executions (batches)
# Replaced complex histogram stats with simple "global_max" tracking per layer.
_CALIBRATION_STATE = {
    "active": False,
    "stats": {},  # {layer_name: float_max_value}
    "step_count": 0,
    "target_steps": 0,
    "hooks_registered": False,
}


def _calibration_max_hook(layer_name):
    """
    Hook to collect the global maximum absolute value of activations.
    This simple MAX strategy is more robust for NVFP4's input_scale which
    must cover the entire dynamic range to prevent block scale saturation.
    """

    def hook(module, input, output):
        if not _CALIBRATION_STATE["active"]:
            return

        # Handle inputs tuple vs tensor
        x = input[0] if isinstance(input, tuple) else input

        # Optimization: Compute Max on GPU, then move single scalar to CPU.
        # This is extremely fast and avoids transferring large tensors.
        batch_max = x.detach().abs().max().item()

        # Update global max for this layer
        current_max = _CALIBRATION_STATE["stats"].get(layer_name, 0.0)
        if batch_max > current_max:
            _CALIBRATION_STATE["stats"][layer_name] = batch_max

    return hook


class DiTFastCalibrationNode:
    """
    Initializes calibration by injecting MAX tracking hooks.
    Replaced unstable percentile logic with robust MAX tracking.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                # Percentile input removed - we always use MAX (100th percentile) for robustness
                "restart": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "Reset & Start New",
                        "label_off": "Continue/Inject",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "setup_calibration"
    CATEGORY = "Quantization/Calibration"

    def setup_calibration(self, model, restart):
        global _CALIBRATION_STATE

        # Locate the diffusion model (Diffusers vs Comfy Native)
        target_model = (
            model.model.diffusion_model
            if hasattr(model.model, "diffusion_model")
            else model.model
        )

        if restart:
            print(
                "[ComfyUI-NVFP4-Quantizer] Resetting calibration statistics (MAX strategy)."
            )
            _CALIBRATION_STATE["active"] = True
            _CALIBRATION_STATE["stats"] = {}
            _CALIBRATION_STATE["step_count"] = 0
            _CALIBRATION_STATE["hooks_registered"] = False

        if not _CALIBRATION_STATE["hooks_registered"]:
            print(
                f"[ComfyUI-NVFP4-Quantizer] Injecting MAX calibration hooks into Linear layers..."
            )
            count = 0
            for name, module in target_model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    clean_name = normalize_layer_name(name)
                    # Register simple max hook
                    module.register_forward_hook(_calibration_max_hook(clean_name))
                    count += 1
            print(f"[ComfyUI-NVFP4-Quantizer] Hooks injected into {count} layers.")
            _CALIBRATION_STATE["hooks_registered"] = True
            _CALIBRATION_STATE["active"] = True
        else:
            # Ensure it's active if it was paused
            _CALIBRATION_STATE["active"] = True
            print(
                f"[ComfyUI-NVFP4-Quantizer] Calibration active (Stats accumulated: {_CALIBRATION_STATE['step_count']} steps so far)."
            )

        return (model,)


class SaveFastCalibrationData:
    """
    Counts execution steps. If step_count >= accumulated_data_points, saves the JSON.
    Uses robust MAX strategy to determine input scales for NVFP4.
    """

    @classmethod
    def INPUT_TYPES(s):
        m_types = list(load_json(MODEL_TYPES_FILENAME).keys())
        return {
            "required": {
                "model": ("MODEL",),  # Passthrough to ensure order
                "model_type": (m_types if m_types else ["None"],),
                "profile_name": ("STRING", {"default": "my_quant_profile"}),
                "accumulate_steps": ("INT", {"default": 1, "min": 1, "max": 1000}),
                "trigger_image": (
                    "IMAGE",
                ),  # Force this node to run after sampler produces image
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "track_and_save"
    OUTPUT_NODE = True
    CATEGORY = "Quantization/Calibration"

    def track_and_save(
        self, model, model_type, profile_name, accumulate_steps, trigger_image
    ):
        global _CALIBRATION_STATE

        ui_text = ""

        if not _CALIBRATION_STATE["active"]:
            ui_text = "Calibration INACTIVE. Connect 'Start Calibration' node."
            print(f"[ComfyUI-NVFP4-Quantizer] {ui_text}")
            return {"ui": {"text": [ui_text]}, "result": (trigger_image,)}

        _CALIBRATION_STATE["step_count"] += 1
        current = _CALIBRATION_STATE["step_count"]

        status_msg = f"Accumulating: {current} / {accumulate_steps}"
        print(f"[ComfyUI-NVFP4-Quantizer] {status_msg}")
        ui_text = status_msg

        if current >= accumulate_steps:
            print(
                "[ComfyUI-NVFP4-Quantizer] Target steps reached. Computing final calibration scales (MAX strategy)..."
            )

            # Calculate final scale for each layer from tracked MAX values
            layer_data = {}

            for layer_name, max_val in _CALIBRATION_STATE["stats"].items():
                if max_val == 0:
                    print(
                        f"[ComfyUI-NVFP4-Quantizer] Warning: Max value is 0 for {layer_name}"
                    )
                    continue

                # Raw Max is the tracked maximum absolute value
                raw_amax = max_val

                # Normalize to comfy-kitchen's NVFP4 scale convention.
                # scale = amax / (F8_E4M3_MAX * F4_E2M1_MAX)
                # NVFP4_SCALE_DIVISOR = 2688.0
                final_scale = raw_amax / NVFP4_SCALE_DIVISOR

                layer_data[layer_name] = float(final_scale)
                # print(f"[ComfyUI-NVFP4-Quantizer] {layer_name}: raw_max={raw_amax:.4f} -> input_scale={final_scale:.6e}")

            # Load existing calibration data
            full_data = load_json(CALIBRATION_FILENAME)

            # Ensure structure: Type -> Profile -> Layers
            if model_type not in full_data:
                full_data[model_type] = {}

            # Update with new calibration data
            full_data[model_type][profile_name] = layer_data

            save_json(full_data, CALIBRATION_FILENAME)

            save_msg = f"✅ Saved to calibration_scales.json\nType: {model_type}\nProfile: {profile_name}\nLayers: {len(layer_data)}"
            print(f"[ComfyUI-NVFP4-Quantizer] {save_msg}")
            ui_text = save_msg

            # Reset/Disable to prevent pollution if user keeps running without reset
            _CALIBRATION_STATE["active"] = False
            _CALIBRATION_STATE["stats"] = {}
            _CALIBRATION_STATE["step_count"] = 0

        return {"ui": {"text": [ui_text]}, "result": (trigger_image,)}


# ============================================================================
# NODE 5: REMOVE RECORD
# ============================================================================


class DiTRemoveRecordNode:
    @classmethod
    def INPUT_TYPES(s):
        # Force reload keys
        m_types = ["None"] + list(load_json(MODEL_TYPES_FILENAME).keys())

        # Presets
        presets = ["None"]
        all_presets = load_json(PRESET_FILENAME)
        for t in all_presets:
            for p in all_presets[t]:
                presets.append(f"{t}::{p}")

        # Calibration profiles (Type::Profile)
        calib_profiles = ["None"]
        calib_data = load_json(CALIBRATION_FILENAME)
        for m_type, profiles in calib_data.items():
            for profile in profiles:
                calib_profiles.append(f"{m_type}::{profile}")

        return {
            "required": {
                "remove_model_type": (m_types,),
                "remove_preset": (presets,),
                "remove_calib_profile": (calib_profiles,),
                "execute": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "DELETE SELECTED",
                        "label_off": "Do Nothing",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status_text",)
    FUNCTION = "remove_record"
    CATEGORY = "Quantization/Ops"
    OUTPUT_NODE = True

    def remove_record(
        self, remove_model_type, remove_preset, remove_calib_profile, execute
    ):
        if not execute:
            return ("Execution not enabled.",)

        log = []

        # 1. Remove Model Type
        if remove_model_type != "None":
            data = load_json(MODEL_TYPES_FILENAME)
            if remove_model_type in data:
                del data[remove_model_type]
                save_json(data, MODEL_TYPES_FILENAME)
                log.append(f"Removed Model Type: {remove_model_type}")
            else:
                log.append(f"Model Type {remove_model_type} not found.")

        # 2. Remove Preset
        if remove_preset != "None" and "::" in remove_preset:
            p_type, p_name = remove_preset.split("::", 1)
            data = load_json(PRESET_FILENAME)
            if p_type in data and p_name in data[p_type]:
                del data[p_type][p_name]
                if not data[p_type]:
                    del data[p_type]
                save_json(data, PRESET_FILENAME)
                log.append(f"Removed Preset: {remove_preset}")
            else:
                log.append(f"Preset {remove_preset} not found.")

        # 3. Remove Calibration Profile
        if remove_calib_profile != "None" and "::" in remove_calib_profile:
            c_type, c_profile = remove_calib_profile.split("::", 1)
            data = load_json(CALIBRATION_FILENAME)
            if c_type in data and c_profile in data[c_type]:
                del data[c_type][c_profile]
                if not data[c_type]:
                    del data[c_type]
                save_json(data, CALIBRATION_FILENAME)
                log.append(f"Removed Calibration Profile: {c_profile} (Type: {c_type})")
            else:
                log.append(f"Profile {c_profile} not found.")

        result_text = "\n".join(log) if log else "Nothing selected to remove."
        print(f"[ComfyUI-NVFP4-Quantizer] {result_text}")
        return (result_text,)


# ============================================================================
# MAPPINGS
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "DiTInspectNode": DiTInspectNode,
    "DiTCompareNode": DiTCompareNode,
    "DiTIsEqualNode": DiTIsEqualNode,
    "DiTQuantizeNode": DiTQuantizeNode,
    "DiTFastCalibrationNode": DiTFastCalibrationNode,
    "SaveFastCalibrationData": SaveFastCalibrationData,
    "DiTRemoveRecordNode": DiTRemoveRecordNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiTInspectNode": "Inspect Model",
    "DiTCompareNode": "Compare Models",
    "DiTIsEqualNode": "Check Model Equality",
    "DiTQuantizeNode": "Quantize Model",
    "DiTFastCalibrationNode": "Start Calibration (Max)",
    "SaveFastCalibrationData": "End Calibration & Save",
    "DiTRemoveRecordNode": "Remove Configuration Record",
}
