import streamlit as st
from PIL import Image, ImageOps
import plotly.graph_objects as go

st.set_page_config(layout="centered", page_title="Fashion Spectrum", page_icon="👗")

# --- 1. モデルとデータの読み込み ---
# Streamlitのキャッシュ機能を使って、アプリケーションの実行中に一度だけリソースを読み込みます。
# これにより、ユーザーがUIを操作するたびに再読み込みされるのを防ぎ、高速化します。
@st.cache_resource
def load_resources():
    """
    アプリケーションに必要なモデル、データ、および特徴量を読み込みます。
    
    Returns:
        tuple: 必要なリソース（デバイス、プロセッサー、モデル、スタイルデータなど）
    """
    print("✅ リソースを読み込み中...")

    import torch
    import torch.nn.functional as F
    from transformers import CLIPProcessor, CLIPModel
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "openai/clip-vit-base-patch32"

    # Hugging FaceからCLIPモデルとプロセッサーを読み込み
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)

    # スタイル提案用のテキストを定義し、ベクトル化
    # UI表示も英語に統一するため、カテゴリ名は英語に変更
    style_categories = {
        "Style": ["streetwear", "vintage", "sporty", "elegant", "preppy", "punk", "gothic", "hippie", "grunge", "y2k"]
    }

    fashion_styles = []
    for category in style_categories.keys():
        fashion_styles.extend(style_categories[category])

    text_inputs = processor(text=fashion_styles, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        style_features = model.get_text_features(**text_inputs)

    print("✅ 読み込み完了！")
    return device, processor, model, fashion_styles, style_categories, style_features, torch, F

def get_resources():
    """
    初回分析時にだけリソースを読み込み、以降は session_state から返す。
    """
    if "resources" not in st.session_state:
        st.session_state["resources"] = load_resources()
    return st.session_state["resources"]

# --- 2. メイン処理のためのヘルパー関数 ---
def calculate_centroid_vector(uploaded_images, weights, device, processor, model):
    """
    アップロードされた画像の重み付け平均（重心）ベクトルを計算します。
    
    Args:
        uploaded_images (list): アップロードされたPIL画像のリスト
        weights (list): 各画像の重要度を示す重みのリスト
        
    Returns:
        torch.Tensor: 計算された重心ベクトル
    """

    import torch
    
    # 重みをテンソルに変換
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    # 重みの合計が0の場合は警告を出して終了
    if weights_tensor.sum() == 0:
        st.warning("All weights are 0. Please set at least one image weight greater than 0.")
        return None

    # 各画像のベクトルを計算
    all_query_features = []
    for image in uploaded_images:
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        all_query_features.append(image_features)

    # 重み付けされたベクトルの合計を計算
    weighted_features = [feat * weight for feat, weight in zip(all_query_features, weights_tensor)]
    weighted_sum = torch.sum(torch.stack(weighted_features), dim=0)

    # 重心ベクトルを計算し、正規化
    query_features_centroid = weighted_sum / weights_tensor.sum()
    query_features_centroid /= query_features_centroid.norm(dim=-1, keepdim=True)
    return query_features_centroid


def display_style_analysis(query_features_centroid, fashion_styles, style_categories, style_features):
    """
    重心ベクトルに基づいて、スタイルの系統やカラーの分析結果をレーダーチャートで表示します。
    
    Args:
        query_features_centroid (torch.Tensor): 計算された重心ベクトル
    """

    import torch.nn.functional as F

    st.header("Analysis Results")
    st.write("We analyzed the attributes that describe your uploaded outfits.")

    for category_name, attributes in style_categories.items():
        st.subheader(category_name)

        labels = []
        scores = []

        # cosine similarity を計算して labels / scores に溜める
        for attribute in attributes:
            try:
                attribute_index = fashion_styles.index(attribute)

                sim = F.cosine_similarity(
                    query_features_centroid,
                    style_features[attribute_index].unsqueeze(0)
                ).item()

                # [-1, 1] → [0, 1] に正規化（レーダー表示しやすい）
                sim01 = (sim + 1.0) / 2.0

                labels.append(attribute)
                scores.append(sim01)

            except ValueError:
                # fashion_styles に無い場合はスキップ
                continue

        if len(scores) == 0:
            st.info("No attributes found for this category.")
            continue

        # radar chart は始点に戻して閉じる必要がある
        labels_closed = labels + [labels[0]]
        scores_closed = scores + [scores[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=scores_closed,
            theta=labels_closed,
            fill='toself'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

# --- 3. Streamlit アプリケーション本体 ---
def main():
    """
    Streamlitアプリケーションのメイン関数。UIの構築と処理の流れを定義します。
    """
    st.title("Fashion Style Spectrum")
    st.write("Decompose outfit images into attributes such as style, color, and silhouette.")

    uploaded_files = st.file_uploader(
        "Upload image(s)...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.markdown("---")
        st.subheader("Uploaded Images & Weighting")

        query_images = []
        weights = []

        n_cols = min(4, len(uploaded_files))
        cols = st.columns(n_cols)

        preview_size = (1000, 1500)  # プレビュー画像のサイズを指定

        # アップロード画像を表示し、重み付けスライダーを配置
        # 列の数を動的に調整、画像を中央クロップして表示
        for i, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file).convert("RGB")
            # 表示用だけ中央クロップしてサイズ統一（元画像imageは変更しない）
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
                    label_visibility="collapsed"
                )
            query_images.append(image)
            weights.append(weight)

        # 分析実行ボタン
        st.markdown("---")
        if st.button("Run analysis"):
            with st.spinner("Loading resources (first time) & analyzing..."):
                # ここで初回だけロードされる
                device, processor, model, fashion_styles, style_categories, style_features, torch, F = get_resources()

                query_features_centroid = calculate_centroid_vector(
                    query_images, weights,
                    device=device, processor=processor, model=model
                )

                if query_features_centroid is not None:
                    st.markdown("---")
                    display_style_analysis(
                        query_features_centroid,
                        fashion_styles=fashion_styles,
                        style_categories=style_categories,
                        style_features=style_features
                    )


# アプリケーションの開始点
if __name__ == "__main__":
    main()
