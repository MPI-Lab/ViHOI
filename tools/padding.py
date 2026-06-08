import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Tuple, Union, Any, List
from tqdm import tqdm

import torch


def setup_logger(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=level,
    )


def find_pt_files(input_dir: Path, recursive: bool = True) -> List[Path]:
    if recursive:
        return sorted([p for p in input_dir.rglob('*.pt') if p.is_file()])
    return sorted([p for p in input_dir.glob('*.pt') if p.is_file()])

# Insert: helper to detect modality from filename suffix

def get_modality_from_filename(file_path: Path) -> str:
    name = file_path.stem.lower()
    if name.endswith('_textual'):
        return 'textual'
    if name.endswith('_visual'):
        return 'visual'
    return ''


def _get_time_and_feat_dims(emb: torch.Tensor) -> Tuple[int, int]:
    if emb.ndim == 2:
        return int(emb.shape[0]), int(emb.shape[1])
    if emb.ndim == 3:
        if int(emb.shape[0]) != 1:
            raise ValueError(f"Expected batch size 1 for 3D embeddings, got {emb.shape}")
        return int(emb.shape[1]), int(emb.shape[2])
    raise ValueError(f"Embedding must be 2D [T, D] or 3D [1, T, D], got shape={tuple(emb.shape)}")


def extract_embedding_and_length(obj: Any) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
    """
    Returns (embedding, length, extra_fields)
    - Supports Tensor of shape [T, D] or [1, T, D]
    - If obj is a dict: expects key 'vlm_embedding' as Tensor, optional 'vlm_length'
      Preserve other keys in extra_fields
    """
    if torch.is_tensor(obj):
        emb = obj
        if emb.ndim < 2:
            raise ValueError(f"Tensor embedding must be [T, D] or [1, T, D], got shape={tuple(emb.shape)}")
        time_len, _ = _get_time_and_feat_dims(emb)
        return emb, int(time_len), {}

    if isinstance(obj, dict):
        if 'vlm_embedding' not in obj:
            raise KeyError("Missing key 'vlm_embedding' in loaded dict")
        emb = obj['vlm_embedding']
        if not torch.is_tensor(emb):
            raise TypeError("'vlm_embedding' must be a torch.Tensor")
        if emb.ndim < 2:
            raise ValueError(f"'vlm_embedding' must be [T, D] or [1, T, D], got shape={tuple(emb.shape)}")
        # Prefer provided length, otherwise infer from shape
        inferred_len, _ = _get_time_and_feat_dims(emb)
        length = int(obj.get('vlm_length', inferred_len))
        extra = {k: v for k, v in obj.items() if k not in {'vlm_embedding', 'vlm_length', 'vlm_mask'}}
        return emb, length, extra

    raise TypeError(f"Unsupported object type: {type(obj)}")


def pad_embedding(embedding: torch.Tensor, target_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads embedding to target_len and returns (padded_embedding, mask[target_len])
    - If input is [T, D], output is [target_len, D]
    - If input is [1, T, D], output is [1, target_len, D]
    - mask is 1D bool with True for valid positions
    """
    if embedding.ndim == 2:
        time_len, feat_dim = _get_time_and_feat_dims(embedding)
        if time_len == target_len:
            mask = torch.zeros(target_len, dtype=torch.bool, device=embedding.device)
            mask[:time_len] = True
            return embedding, mask
        padded = torch.zeros(target_len, feat_dim, dtype=embedding.dtype, device=embedding.device)
        padded[:time_len] = embedding
        mask = torch.zeros(target_len, dtype=torch.bool, device=embedding.device)
        mask[:time_len] = True
        return padded, mask

    if embedding.ndim == 3:
        if int(embedding.shape[0]) != 1:
            raise ValueError(f"Expected batch size 1 for 3D embeddings, got {tuple(embedding.shape)}")
        time_len, feat_dim = _get_time_and_feat_dims(embedding)
        if time_len == target_len:
            mask = torch.zeros(target_len, dtype=torch.bool, device=embedding.device)
            mask[:time_len] = True
            return embedding, mask
        padded = torch.zeros(1, target_len, feat_dim, dtype=embedding.dtype, device=embedding.device)
        padded[:, :time_len] = embedding
        mask = torch.zeros(target_len, dtype=torch.bool, device=embedding.device)
        mask[:time_len] = True
        return padded, mask

    raise ValueError(f"Unsupported embedding ndim: {embedding.ndim}")


def infer_feature_dim(files: List[Path]) -> int:
    feat_dim: Union[int, None] = None
    for f in files:
        obj = torch.load(f, map_location='cpu')
        emb, length, _ = extract_embedding_and_length(obj)
        _, cur_dim = _get_time_and_feat_dims(emb)
        if feat_dim is None:
            feat_dim = cur_dim
        elif feat_dim != cur_dim:
            raise ValueError(f"Feature dim mismatch: {f} has D={cur_dim}, expected D={feat_dim}")
    assert feat_dim is not None
    return feat_dim


def compute_max_length(files: List[Path]) -> int:
    max_len = 0
    for f in files:
        obj = torch.load(f, map_location='cpu')
        emb, length, _ = extract_embedding_and_length(obj)
        max_len = max(max_len, int(length))
    if max_len <= 0:
        raise ValueError("Max length computed as 0; no valid sequences found")
    return max_len


def make_output_path(input_root: Path, output_root: Path, file_path: Path) -> Path:
    rel = file_path.relative_to(input_root)
    out_path = output_root / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


# Replace: make processing per-modality (textual / visual) instead of global

def process_directory(input_dir: Path, output_dir: Path, recursive: bool) -> None:
    files = find_pt_files(input_dir, recursive=recursive)
    if not files:
        logging.error(f"No .pt files found under {input_dir}")
        return

    # Group files by modality based on filename suffix
    modality_to_files: Dict[str, List[Path]] = {"textual": [], "visual": []}
    skipped_files: List[Path] = []
    for f in files:
        modality = get_modality_from_filename(f)
        if modality in modality_to_files:
            modality_to_files[modality].append(f)
        else:
            skipped_files.append(f)

    if skipped_files:
        logging.warning(f"Skipping {len(skipped_files)} files without '_textual' or '_visual' suffix")

    # Compute per-modality stats
    modality_stats: Dict[str, Dict[str, int]] = {}
    for modality, m_files in modality_to_files.items():
        if not m_files:
            logging.warning(f"No {modality} files found under {input_dir}")
            continue
        logging.info(f"Scanning {len(m_files)} {modality} files to determine max length and feature dim...")
        feat_dim = infer_feature_dim(m_files)
        max_len = compute_max_length(m_files)
        modality_stats[modality] = {"feat_dim": feat_dim, "max_len": max_len}
        logging.info(f"{modality.capitalize()} settings: feature_dim={feat_dim}, max_length={max_len}")

    if not modality_stats:
        logging.error("No eligible files to process.")
        return

    num_ok, num_error = 0, 0
    # Preserve deterministic order: textual then visual (each sorted already by find_pt_files)
    files_to_process: List[Path] = [*modality_to_files.get('textual', []), *modality_to_files.get('visual', [])]
    for f in tqdm(files_to_process):
        modality = get_modality_from_filename(f)
        stats = modality_stats.get(modality)
        if not stats:
            continue
        try:
            obj = torch.load(f, map_location='cpu')
            emb, length, extra = extract_embedding_and_length(obj)
            _, cur_dim = _get_time_and_feat_dims(emb)

            expected_dim = stats['feat_dim']
            target_len = stats['max_len']
            if int(cur_dim) != expected_dim:
                raise ValueError(f"Feature dim mismatch in {f}: {cur_dim} vs {expected_dim} (modality={modality})")

            padded, mask = pad_embedding(emb, target_len)

            # Build output object, preserving extra fields
            out_obj: Dict[str, Any] = dict(extra)
            out_obj['vlm_embedding'] = padded.contiguous()
            out_obj['vlm_mask'] = mask
            out_obj['vlm_length'] = int(length)

            out_path = make_output_path(input_dir, output_dir, f)
            torch.save(out_obj, out_path)
            num_ok += 1
        except Exception as e:
            logging.exception(f"Failed to process {f}: {e}")
            num_error += 1

    logging.info(f"Done. Succeeded: {num_ok}, Failed: {num_error}. Output at {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pad VLM embeddings to global max length and create masks.")
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing .pt files (variable-length embeddings)')
    parser.add_argument('--output-dir', type=str, default='',
                        help='Directory to write padded files. Default: <input-dir>_padded')
    parser.add_argument('--no-recursive', action='store_true',
                        help='Do not search subdirectories')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='Increase verbosity (-v, -vv)')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(args.verbose)

    input_dir = Path(args.input_dir).expanduser().resolve()
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        # Insert '_padded' after the 'hidden_<num>' part in the directory name
        input_dir_str = str(input_dir)
        import re
        # Match 'hidden_<num>' and insert '_padded' after it
        output_dir_str = re.sub(r'(hidden_\d+)', r'\1_padded', input_dir_str)
        output_dir = Path(output_dir_str)
    

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Input: {input_dir}")
    logging.info(f"Output: {output_dir}")

    process_directory(input_dir=input_dir, output_dir=output_dir, recursive=(not args.no_recursive))
    # python mytest/padding.py --input-dir ./processed_data/20250921-082501-gen3_vlm_hidden_12_test -v

if __name__ == '__main__':
    main()
