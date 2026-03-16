#!/usr/bin/env python3
"""
laion_fashion_quantile_prototype.py

Style Spectrum のメインコードに接続できるようにした、
LAION-RVS-Fashion 用の分位点取得プロトタイプ。

目的
- 画像をローカル保存せずに URL から逐次取得する
- メインコードと同じ軸定義 JSON・同じテキスト特徴量生成ロジックで raw score を計算する
- Basic は対極属性の差分スコアを集計する
- Culture は各属性の sim01 スコアを集計する
- attribute_norm_stats にそのまま流し込みやすい JSON を出力する

出力例
{
  "formal_vs_casual": {"min": 0.001, "max": 0.048},
  "classic_vs_modern": {"min": 0.002, "max": 0.052},
  "streetwear": {"min": 0.591, "max": 0.638},
  ...
}

依存関係
- torch
- transformers
- datasets
- pillow
- requests
- numpy

実行例
python laion_fashion_quantile_prototype.py \
  --config config/style_config.json \
  --output-dir outputs/laion_quantiles \
  --max-samples 20000 \
  --types COMPLEX PARTIAL_COMPLEX \
  --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import signal
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import requests
from datasets import load_dataset
from PIL import Image, UnidentifiedImageError


# --------------------------------------------------
# 外部JSON読み込み
# --------------------------------------------------


def load_style_config(config_path: Path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------
# Utilities
# --------------------------------------------------


def log(msg: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


class GracefulStop:
    def __init__(self) -> None:
        self.stop_requested = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame) -> None:  # type: ignore[no-untyped-def]
        self.stop_requested = True
        log(f"Stop requested by signal {signum}; saving progress soon.")


class RunningStore:
    def __init__(self) -> None:
        self.values: List[float] = []

    def update(self, value: float) -> None:
        self.values.append(float(value))

    def summary(self, lower_q: float, upper_q: float) -> Dict[str, float]:
        if len(self.values) == 0:
            return {
                "count": 0,
                "min": float("nan"),
                "max": float("nan"),
                "p01": float("nan"),
                "p99": float("nan"),
                "mean": float("nan"),
                "std": float("nan"),
            }

        arr = np.asarray(self.values, dtype=np.float32)
        return {
            "count": int(arr.size),
            "min": float(np.percentile(arr, lower_q)),
            "max": float(np.percentile(arr, upper_q)),
            "p01": float(np.percentile(arr, 1)),
            "p99": float(np.percentile(arr, 99)),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        }


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------
# メインコード互換の特徴量生成
# --------------------------------------------------


def load_resources(style_config, device_arg: Optional[str] = None):
    import torch
    import torch.nn.functional as F
    from transformers import CLIPModel, CLIPProcessor

    device = device_arg or ("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "openai/clip-vit-base-patch32"

    log(f"Loading CLIP resources: {model_name} on {device}")

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()

    style_categories = style_config["style_categories"]

    fashion_styles = []
    attribute_prompt_map = {}
    for category in style_categories.keys():
        for attribute, prompts in style_categories[category].items():
            fashion_styles.append(attribute)
            attribute_prompt_map[attribute] = prompts

    style_features_list = []
    for attribute in fashion_styles:
        text_inputs = processor(
            text=attribute_prompt_map[attribute],
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            prompt_features = model.get_text_features(**text_inputs)

        prompt_features = prompt_features / prompt_features.norm(dim=-1, keepdim=True)
        attribute_feature = prompt_features.mean(dim=0, keepdim=True)
        attribute_feature = attribute_feature / attribute_feature.norm(dim=-1, keepdim=True)
        style_features_list.append(attribute_feature.squeeze(0))

    style_features = torch.stack(style_features_list)

    return (
        device,
        processor,
        model,
        fashion_styles,
        style_categories,
        style_features,
        torch,
        F,
    )


# --------------------------------------------------
# スコア計算（メインコード互換）
# --------------------------------------------------


def compute_attribute_scores_from_image(
    image,
    device,
    processor,
    model,
    fashion_styles,
    style_features,
    style_categories,
    basic_opposite_pairs,
    F,
    torch,
):
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    raw_score_map = {}
    for attribute in fashion_styles:
        attribute_index = fashion_styles.index(attribute)
        sim = F.cosine_similarity(
            image_features,
            style_features[attribute_index].unsqueeze(0),
        ).item()
        sim01 = (sim + 1.0) / 2.0
        raw_score_map[attribute] = float(sim01)

    basic_diff_score_map = {}
    for left, right in basic_opposite_pairs:
        left_score = raw_score_map.get(left, 0.0)
        right_score = raw_score_map.get(right, 0.0)
        diff_score = abs(left_score - right_score)
        basic_diff_score_map[f"{left}_vs_{right}"] = float(diff_score)

    culture_score_map = {}
    for attribute in style_categories["Culture"].keys():
        culture_score_map[attribute] = raw_score_map[attribute]

    return raw_score_map, basic_diff_score_map, culture_score_map


# --------------------------------------------------
# Dataset / image fetching
# --------------------------------------------------


def stream_rows(split: str) -> Iterable[dict]:
    ds = load_dataset("Slep/LAION-RVS-Fashion", split=split, streaming=True)
    return ds


class ImageFetcher:
    def __init__(self, timeout: float = 15.0, user_agent: str = "style-spectrum-quantile-prototype/0.2") -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def fetch(self, url: str):
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except (requests.RequestException, UnidentifiedImageError, OSError):
            return None


# --------------------------------------------------
# Filtering
# --------------------------------------------------


def should_keep_row(
    row: dict,
    allowed_types: Optional[set],
    allowed_categories: Optional[set],
    max_punsafe: float,
    max_pwatermark: float,
    min_width: int,
    min_height: int,
    dedupe_products: bool,
    seen_products: set,
) -> bool:
    row_type = row.get("TYPE")
    row_category = row.get("CATEGORY")
    punsafe = float(row.get("punsafe", 1.0) or 1.0)
    pwatermark = float(row.get("pwatermark", 1.0) or 1.0)
    width = int(float(row.get("WIDTH", 0) or 0))
    height = int(float(row.get("HEIGHT", 0) or 0))

    if allowed_types is not None and row_type not in allowed_types:
        return False
    if allowed_categories is not None and row_category not in allowed_categories:
        return False
    if punsafe > max_punsafe:
        return False
    if pwatermark > max_pwatermark:
        return False
    if width < min_width or height < min_height:
        return False

    if dedupe_products:
        product_id = row.get("PRODUCT_ID")
        if product_id is not None:
            try:
                pid = int(product_id)
            except (TypeError, ValueError):
                pid = None
            if pid is not None:
                if pid in seen_products:
                    return False
                seen_products.add(pid)

    return True


# --------------------------------------------------
# Main
# --------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--types", nargs="*", default=["COMPLEX", "PARTIAL_COMPLEX"])
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--max-punsafe", type=float, default=0.5)
    parser.add_argument("--max-pwatermark", type=float, default=0.5)
    parser.add_argument("--min-width", type=int, default=256)
    parser.add_argument("--min-height", type=int, default=256)
    parser.add_argument("--dedupe-products", action="store_true")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    style_config = load_style_config(args.config)
    style_categories = style_config["style_categories"]
    basic_opposite_pairs = style_config["basic_opposite_pairs"]
    lower_percentile = style_config["quantile_settings"]["lower_percentile"]
    upper_percentile = style_config["quantile_settings"]["upper_percentile"]

    project_root = Path(__file__).resolve().parent.parent
    data_output_path = project_root / "data" / "attribute_norm_stats_for_app.json"

    stopper = GracefulStop()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    data_output_path.parent.mkdir(parents=True, exist_ok=True)

    output_attribute_stats_path = output_dir / "attribute_norm_stats_for_app.json"

    (
        device,
        processor,
        model,
        fashion_styles,
        style_categories,
        style_features,
        torch,
        F,
    ) = load_resources(style_config=style_config, device_arg=args.device)

    fetcher = ImageFetcher(timeout=args.timeout)

    basic_keys = [f"{left}_vs_{right}" for left, right in basic_opposite_pairs]
    culture_keys = list(style_categories["Culture"].keys())
    all_stat_keys = basic_keys + culture_keys
    stores = {key: RunningStore() for key in all_stat_keys}

    allowed_types = set(args.types) if args.types else None
    allowed_categories = set(args.categories) if args.categories else None
    seen_products = set()

    counters = {
        "rows_seen": 0,
        "rows_kept": 0,
        "download_ok": 0,
        "download_failed": 0,
        "scored": 0,
    }

    counts_per_source: Dict[str, int] = defaultdict(int)

    write_json(
        output_dir / "job_meta.json",
        {
            "dataset": "Slep/LAION-RVS-Fashion",
            "split": args.split,
            "max_samples": args.max_samples,
            "types": args.types,
            "categories": args.categories,
            "lower_percentile": lower_percentile,
            "upper_percentile": upper_percentile,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    failures_fp = (output_dir / "failed_urls.jsonl").open("a", encoding="utf-8")
    sampled_fp = (output_dir / "sampled_rows.jsonl").open("a", encoding="utf-8")

    try:
        for row in stream_rows(args.split):
            if stopper.stop_requested or counters["scored"] >= args.max_samples:
                break

            counters["rows_seen"] += 1

            keep = should_keep_row(
                row=row,
                allowed_types=allowed_types,
                allowed_categories=allowed_categories,
                max_punsafe=args.max_punsafe,
                max_pwatermark=args.max_pwatermark,
                min_width=args.min_width,
                min_height=args.min_height,
                dedupe_products=args.dedupe_products,
                seen_products=seen_products,
            )
            if not keep:
                continue

            counters["rows_kept"] += 1

            url = row.get("URL")
            if not url:
                continue

            image = fetcher.fetch(url)
            if image is None:
                counters["download_failed"] += 1
                failures_fp.write(
                    json.dumps(
                        {
                            "url": url,
                            "index_src": row.get("INDEX_SRC"),
                        },
                        ensure_ascii=False,
                    ) + "\n"
                )
                continue

            counters["download_ok"] += 1
            counts_per_source[str(row.get("INDEX_SRC", "UNKNOWN"))] += 1

            raw_score_map, basic_diff_score_map, culture_score_map = compute_attribute_scores_from_image(
                image=image,
                device=device,
                processor=processor,
                model=model,
                fashion_styles=fashion_styles,
                style_features=style_features,
                style_categories=style_categories,
                basic_opposite_pairs=basic_opposite_pairs,
                F=F,
                torch=torch,
            )

            for key, value in basic_diff_score_map.items():
                stores[key].update(value)
            for key, value in culture_score_map.items():
                stores[key].update(value)

            sampled_fp.write(
                json.dumps(
                    {
                        "id": stable_hash(url),
                        "url": url,
                        "index_src": row.get("INDEX_SRC"),
                        "type": row.get("TYPE"),
                        "category": row.get("CATEGORY"),
                        "product_id": row.get("PRODUCT_ID"),
                        "scores": {
                            "basic_diff": basic_diff_score_map,
                            "culture": culture_score_map,
                        },
                    },
                    ensure_ascii=False,
                ) + "\n"
            )

            counters["scored"] += 1

            if counters["scored"] % args.save_every == 0:
                quantile_payload = {
                    key: {
                        "min": stores[key].summary(lower_percentile, upper_percentile)["min"],
                        "max": stores[key].summary(lower_percentile, upper_percentile)["max"],
                    }
                    for key in all_stat_keys
                }
                write_json(output_attribute_stats_path, quantile_payload)
                write_json(data_output_path, quantile_payload)
                write_json(
                    output_dir / "running_summary.json",
                    {
                        "counters": counters,
                        "attributes": {
                            key: stores[key].summary(lower_percentile, upper_percentile)
                            for key in all_stat_keys
                        },
                        "counts_per_source_top20": dict(
                            sorted(counts_per_source.items(), key=lambda kv: kv[1], reverse=True)[:20]
                        ),
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )
                log(
                    f"scored={counters['scored']} rows_seen={counters['rows_seen']} "
                    f"download_failed={counters['download_failed']}"
                )

        final_attribute_norm_stats = {
            key: {
                "min": stores[key].summary(lower_percentile, upper_percentile)["min"],
                "max": stores[key].summary(lower_percentile, upper_percentile)["max"],
            }
            for key in all_stat_keys
        }
        final_summary = {
            "counters": counters,
            "attributes": {
                key: stores[key].summary(lower_percentile, upper_percentile)
                for key in all_stat_keys
            },
            "counts_per_source_top50": dict(
                sorted(counts_per_source.items(), key=lambda kv: kv[1], reverse=True)[:50]
            ),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        write_json(output_attribute_stats_path, final_attribute_norm_stats)
        write_json(data_output_path, final_attribute_norm_stats)
        write_json(output_dir / "final_summary.json", final_summary)
        log(
            f"Finished. Wrote {data_output_path}, {output_attribute_stats_path}, "
            f"and {output_dir / 'final_summary.json'}"
        )

    finally:
        failures_fp.close()
        sampled_fp.close()


if __name__ == "__main__":
    main()
