from pathlib import Path
import json

import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm


# ==================================================
# 設定
# ==================================================

IMAGE_DIR = "./fashion_images"
OUTPUT_DIR = "./quantile_output"
BATCH_SIZE = 64
MODEL_NAME = "openai/clip-vit-base-patch32"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ==================================================
# スタイル定義（元コードに合わせる）
# ==================================================

style_categories = {
    "Basic": {
        "formal": [
            "formal fashion",
            "formal outfit",
            "tailored style",
        ],
        "classic": [
            "classic fashion",
            "classic outfit",
            "timeless style",
        ],
        "minimalist": [
            "minimalist fashion",
            "minimal outfit",
            "simple clean style",
        ],
        "monochrome": [
            "monochrome fashion",
            "black and white outfit",
            "single color style",
        ],
        "casual": [
            "casual fashion",
            "casual outfit",
            "everyday relaxed style",
        ],
        "modern": [
            "modern fashion",
            "modern outfit",
            "contemporary style",
        ],
        "detailed": [
            "detailed fashion",
            "decorative outfit",
            "intricate styling",
        ],
        "colorful": [
            "colorful fashion",
            "colorful outfit",
            "vivid colorful style",
        ],
    },
    "Culture": {
        "streetwear": [
            "streetwear",
            "street wear",
            "street outfit",
            "street outfits",
        ],
        "vintage": [
            "vintage fashion",
            "retro outfit",
            "vintage outfit",
        ],
        "sporty": [
            "sporty fashion",
            "sporty outfit",
            "athletic casual style",
        ],
        "elegant": [
            "elegant fashion",
            "elegant outfit",
            "refined graceful style",
        ],
        "preppy": [
            "preppy fashion",
            "preppy outfit",
            "ivy league style",
        ],
        "punk": [
            "punk fashion",
            "punk outfit",
            "rebellious punk style",
        ],
        "gothic": [
            "gothic fashion",
            "gothic outfit",
            "dark gothic style",
        ],
        "hippie": [
            "hippie fashion",
            "bohemian outfit",
            "free spirited hippie style",
        ],
        "grunge": [
            "grunge fashion",
            "grunge outfit",
            "90s grunge style",
        ],
        "y2k": [
            "y2k fashion",
            "2000s inspired outfit",
            "y2k outfit",
        ],
    },
}

opposite_pairs = [
    ("formal", "casual"),
    ("classic", "modern"),
    ("colorful", "monochrome"),
    ("detailed", "minimalist"),
]


# ==================================================
# ヘルパー
# ==================================================

def build_attribute_prompt_map(style_categories):
    fashion_styles = []
    attribute_prompt_map = {}

    for category in style_categories.keys():
        for attribute, prompts in style_categories[category].items():
            fashion_styles.append(attribute)
            attribute_prompt_map[attribute] = prompts

    return fashion_styles, attribute_prompt_map


def build_text_features(processor, model, device, fashion_styles, attribute_prompt_map):
    """
    元コードと同じ方法で、
    複数プロンプト -> 正規化 -> 平均 -> 再正規化
    により各属性ベクトルを作る
    """
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
    return style_features


def list_image_paths(image_dir):
    image_dir = Path(image_dir)
    paths = [p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    return sorted(paths)


def load_images(paths):
    images = []
    valid_paths = []

    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
            valid_paths.append(str(p))
        except Exception as e:
            print(f"skip: {p} ({e})")

    return images, valid_paths


def compute_image_features_in_batches(paths, processor, model, device, batch_size=64):
    all_features = []
    all_valid_paths = []

    for i in tqdm(range(0, len(paths), batch_size), desc="Encoding images"):
        batch_paths = paths[i:i + batch_size]
        images, valid_paths = load_images(batch_paths)

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        all_features.append(image_features.cpu())
        all_valid_paths.extend(valid_paths)

    if len(all_features) == 0:
        raise RuntimeError("有効な画像が1枚も読み込めませんでした。")

    all_features = torch.cat(all_features, dim=0)
    return all_features, all_valid_paths


def compute_attribute_scores(image_features, style_features):
    """
    cosine similarity -> sim01 に変換
    image_features: [N, dim] (normalized)
    style_features: [M, dim] (normalized)
    戻り値: [N, M] の sim01
    """
    sims = image_features @ style_features.T.cpu()
    sims01 = (sims + 1.0) / 2.0
    return sims01.numpy()


def quantile_dict_from_scores(scores_2d, fashion_styles, q_low=0.01, q_high=0.99):
    result = {}
    for i, attribute in enumerate(fashion_styles):
        col = scores_2d[:, i]
        result[attribute] = {
            "min": float(np.quantile(col, q_low)),
            "max": float(np.quantile(col, q_high)),
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "count": int(len(col)),
        }
    return result


def basic_diff_quantiles_for_app(scores_2d, fashion_styles, pairs, q_low=0.01, q_high=0.99):
    """
    アプリ側で使うキー形式:
      formal_vs_casual
      classic_vs_modern
      ...
    で、その差分スコアの1%,99%を返す
    """
    idx_map = {attr: i for i, attr in enumerate(fashion_styles)}
    result = {}

    for left, right in pairs:
        left_scores = scores_2d[:, idx_map[left]]
        right_scores = scores_2d[:, idx_map[right]]

        diff_scores = np.abs(left_scores - right_scores)

        result[f"{left}_vs_{right}"] = {
            "min": float(np.quantile(diff_scores, q_low)),
            "max": float(np.quantile(diff_scores, q_high)),
            "mean": float(np.mean(diff_scores)),
            "std": float(np.std(diff_scores)),
            "count": int(len(diff_scores)),
        }

    return result


def build_attribute_norm_stats_for_app(attribute_stats, basic_diff_stats):
    """
    Streamlit側の attribute_norm_stats にそのまま使いやすい形にする

    - Basic: formal_vs_casual などの差分キー
    - Culture: streetwear など各属性キー
    """
    result = {}

    for key, value in basic_diff_stats.items():
        result[key] = {
            "min": value["min"],
            "max": value["max"],
        }

    for attribute in style_categories["Culture"].keys():
        value = attribute_stats[attribute]
        result[attribute] = {
            "min": value["min"],
            "max": value["max"],
        }

    return result


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    fashion_styles, attribute_prompt_map = build_attribute_prompt_map(style_categories)

    print("Building text features...")
    style_features = build_text_features(
        processor=processor,
        model=model,
        device=device,
        fashion_styles=fashion_styles,
        attribute_prompt_map=attribute_prompt_map,
    )

    image_paths = list_image_paths(IMAGE_DIR)
    print(f"found images: {len(image_paths)}")

    if len(image_paths) == 0:
        raise RuntimeError(f"画像が見つかりません: {IMAGE_DIR}")

    image_features, valid_paths = compute_image_features_in_batches(
        image_paths,
        processor=processor,
        model=model,
        device=device,
        batch_size=BATCH_SIZE,
    )

    print(f"valid images: {len(valid_paths)}")

    print("Computing attribute scores...")
    scores_2d = compute_attribute_scores(
        image_features=image_features,
        style_features=style_features,
    )

    attribute_stats = quantile_dict_from_scores(
        scores_2d=scores_2d,
        fashion_styles=fashion_styles,
        q_low=0.01,
        q_high=0.99,
    )

    basic_diff_stats = basic_diff_quantiles_for_app(
        scores_2d=scores_2d,
        fashion_styles=fashion_styles,
        pairs=opposite_pairs,
        q_low=0.01,
        q_high=0.99,
    )

    attribute_norm_stats_for_app = build_attribute_norm_stats_for_app(
        attribute_stats=attribute_stats,
        basic_diff_stats=basic_diff_stats,
    )

    per_image_scores = []
    for path, row in zip(valid_paths, scores_2d):
        item = {"image_path": path}
        for attr, score in zip(fashion_styles, row):
            item[attr] = float(score)
        per_image_scores.append(item)

    save_json(attribute_stats, Path(OUTPUT_DIR) / "attribute_quantiles.json")
    save_json(basic_diff_stats, Path(OUTPUT_DIR) / "basic_diff_quantiles.json")
    save_json(attribute_norm_stats_for_app, Path(OUTPUT_DIR) / "attribute_norm_stats_for_app.json")
    save_json(per_image_scores, Path(OUTPUT_DIR) / "per_image_scores.json")

    print("\n=== attribute_norm_stats_for_app.json ===")
    print(json.dumps(attribute_norm_stats_for_app, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()