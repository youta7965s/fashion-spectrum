import streamlit as st
from PIL import Image, ImageOps
import plotly.graph_objects as go
import json
from pathlib import Path

st.set_page_config(
    layout="centered",
    page_title="Style Spectrum",
    page_icon="🕸️"
)

# --------------------------------------------------
#  外部JSON読み込み
# --------------------------------------------------

STYLE_CONFIG_PATH = Path("config/style_config.json")
ATTRIBUTE_NORM_STATS_PATH = Path("data/attribute_norm_stats_for_app.json")


def load_style_config():
    with open(STYLE_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_attribute_norm_stats():
    if ATTRIBUTE_NORM_STATS_PATH.exists():
        with open(ATTRIBUTE_NORM_STATS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# --------------------------------------------------
#  モデルとデータの読み込み
# --------------------------------------------------

@st.cache_resource
def load_resources():
    """
    必要なモデル・データを読み込む（初回のみ実行）
    """

    print("✅ リソースを読み込み中...")

    import torch
    import torch.nn.functional as F
    from transformers import CLIPProcessor, CLIPModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "openai/clip-vit-base-patch32"

    # CLIPモデル読み込み
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)

    style_config = load_style_config()

    # スタイルカテゴリ
    style_categories = style_config["style_categories"]

    fashion_styles = []
    attribute_prompt_map = {}
    for category in style_categories.keys():
        for attribute, prompts in style_categories[category].items():
            fashion_styles.append(attribute)
            attribute_prompt_map[attribute] = prompts

    # テキスト特徴量生成
    style_features_list = []
    for attribute in fashion_styles:
        text_inputs = processor(
            text=attribute_prompt_map[attribute],
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            prompt_features = model.get_text_features(**text_inputs)

        prompt_features = prompt_features / prompt_features.norm(dim=-1, keepdim=True)
        attribute_feature = prompt_features.mean(dim=0, keepdim=True)
        attribute_feature = attribute_feature / attribute_feature.norm(dim=-1, keepdim=True)

        style_features_list.append(attribute_feature.squeeze(0))

    style_features = torch.stack(style_features_list)

    print("✅ 読み込み完了！")

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


def get_resources():
    """
    session_state を使い、初回のみロード
    """
    if "resources" not in st.session_state:
        st.session_state["resources"] = load_resources()

    return st.session_state["resources"]


# --------------------------------------------------
#  ヘルパー関数
# --------------------------------------------------

def calculate_centroid_vector(uploaded_images, weights, device, processor, model):
    """
    アップロード画像の重み付き平均ベクトル（重心）を計算
    """

    import torch

    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    # 重みチェック
    if weights_tensor.sum() == 0:
        st.warning(
            "All weights are 0. Please set at least one image weight greater than 0."
        )
        return None

    # 各画像の特徴ベクトル
    all_query_features = []
    for image in uploaded_images:
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)

        all_query_features.append(image_features)

    # 重み付き合計
    weighted_features = [
        feat * weight
        for feat, weight in zip(all_query_features, weights_tensor)
    ]

    weighted_sum = torch.sum(torch.stack(weighted_features), dim=0)

    # 重心ベクトル
    query_features_centroid = weighted_sum / weights_tensor.sum()
    query_features_centroid /= query_features_centroid.norm(
        dim=-1,
        keepdim=True
    )

    return query_features_centroid

def normalize_score_minmax(value, vmin, vmax, eps=1e-12):
    """
    value を [vmin, vmax] で min-max 正規化して [0,1] にする。
    クリップも行う（レンジ外でも崩れない）。
    """
    if vmin is None or vmax is None:
        return value  # statsが無ければそのまま返す

    denom = (vmax - vmin)
    if abs(denom) < eps:
        # min==max 付近（分布が潰れてる）なら真ん中扱い
        return 0.5

    norm = (value - vmin) / denom
    # 0〜1にクリップ（1%/99%設計と相性がいい）
    return max(0.0, min(1.0, norm))

def display_style_analysis(
    query_features_centroid,
    fashion_styles,
    style_categories,
    style_features,
    attribute_norm_stats=None,
):
    """
    スタイル分析結果をレーダーチャートで表示
    """

    import torch.nn.functional as F

    # None対策（呼び出し側が渡さなくても落ちない）
    if attribute_norm_stats is None:
        attribute_norm_stats = {}

    style_config = load_style_config()
    opposite_pairs = style_config["basic_opposite_pairs"]

    for category_name, attributes in style_categories.items():

        st.subheader(category_name)

        labels = []
        scores = []
        raw_scores = []

        # cosine similarity
        for attribute in attributes:
            try:
                attribute_index = fashion_styles.index(attribute)

                sim = F.cosine_similarity(
                    query_features_centroid,
                    style_features[attribute_index].unsqueeze(0),
                ).item()

                # [-1,1] → [0,1]
                sim01 = (sim + 1.0) / 2.0

                if category_name == "Basic":
                    labels.append(attribute)
                    raw_scores.append(sim01)
                else:
                    # attributeごとのmin/maxで再正規化（1%/99%想定）
                    stat = attribute_norm_stats.get(attribute, {})
                    vmin = stat.get("min")
                    vmax = stat.get("max")
                    score_norm = normalize_score_minmax(sim01, vmin, vmax)

                    labels.append(attribute)
                    scores.append(score_norm)

            except ValueError:
                continue

        if category_name == "Basic":
            if len(raw_scores) == 0:
                st.info("No attributes found for this category.")
                continue

            score_map = dict(zip(labels, raw_scores))

            plot_labels = []
            plot_scores = []

            for left, right in opposite_pairs:
                left_score = score_map.get(left, 0.0)
                right_score = score_map.get(right, 0.0)

                if left_score >= right_score:
                    diff_score = left_score - right_score
                    stat = attribute_norm_stats.get(f"{left}_vs_{right}", {})
                    vmin = stat.get("min")
                    vmax = stat.get("max")
                    plot_labels.append(left)
                    plot_scores.append(
                        normalize_score_minmax(diff_score, vmin, vmax)
                    )
                else:
                    diff_score = right_score - left_score
                    stat = attribute_norm_stats.get(f"{left}_vs_{right}", {})
                    vmin = stat.get("min")
                    vmax = stat.get("max")
                    plot_labels.append(right)
                    plot_scores.append(
                        normalize_score_minmax(diff_score, vmin, vmax)
                    )

            # radar chart は閉じる必要がある
            labels_closed = plot_labels + [plot_labels[0]]
            scores_closed = plot_scores + [plot_scores[0]]

            fig = go.Figure()

            fig.add_trace(
                go.Scatterpolar(
                    r=scores_closed,
                    theta=labels_closed,
                    fill="toself",
                    line_shape="spline",
                )
            )

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                    ),
                    angularaxis=dict(
                        categoryorder="array",
                        categoryarray=labels,
                    ),
                ),
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20),
            )

            st.plotly_chart(fig, use_container_width=True)

        elif category_name == "Culture":
            if len(scores) == 0:
                st.info("No attributes found for this category.")
                continue

            ranked_items = sorted(
                zip(labels, scores),
                key=lambda x: x[1],
                reverse=True,
            )[:3]

            for label, score in ranked_items:
                st.write(label)
                st.progress(float(score))


# --------------------------------------------------
#  Streamlit アプリ本体
# --------------------------------------------------

def main():

    st.title("Style Spectrum")

    uploaded_files = st.file_uploader(
        "Upload image(s)...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Max 5 images. Clear background recommended.",
    )

    if uploaded_files:

        st.markdown("---")
        st.subheader("Images & Weighting")

        query_images = []
        weights = []

        n_cols = min(4, len(uploaded_files))
        cols = st.columns(n_cols)

        preview_size = (1000, 1500)

        # 画像表示 + 重み付け
        for i, uploaded_file in enumerate(uploaded_files):

            image = Image.open(uploaded_file).convert("RGB")

            preview = ImageOps.fit(
                image,
                preview_size,
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )

            with cols[i % n_cols]:

                st.image(preview)

                weight = st.slider(
                    label="",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    key=f"slider_{uploaded_file.name}",
                    label_visibility="collapsed",
                )

            query_images.append(image)
            weights.append(weight)

        # 外部から渡すmin/max（将来的にDB由来に置き換える）
        attribute_norm_stats = load_attribute_norm_stats()

        # 分析実行
        if st.button("Run analysis"):

            with st.spinner(
                "Loading resources (first time) & analyzing..."
            ):

                device, processor, model, fashion_styles, style_categories, style_features, torch, F = get_resources()

                query_features_centroid = calculate_centroid_vector(
                    query_images,
                    weights,
                    device=device,
                    processor=processor,
                    model=model,
                )

                if query_features_centroid is not None:

                    st.markdown("---")

                    display_style_analysis(
                        query_features_centroid,
                        fashion_styles=fashion_styles,
                        style_categories=style_categories,
                        style_features=style_features,
                        attribute_norm_stats=attribute_norm_stats,
                    )


# --------------------------------------------------
# entry point
# --------------------------------------------------

if __name__ == "__main__":
    main()