import streamlit as st
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "openai/clip-vit-base-patch32"

    # Hugging FaceからCLIPモデルとプロセッサーを読み込み
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)

    # スタイル提案用のテキストを定義し、ベクトル化
    # UI表示も英語に統一するため、カテゴリ名は英語に変更
    style_categories = {
        "Style": ["streetwear", "vintage", "modern", "sporty", "elegant", "preppy", "minimalist", "punk", "gothic", "hippie", "grunge"],
        "Color": ["red", "blue", "green", "yellow", "black", "white", "pink", "purple", "orange", "brown", "gray"]
    }

    fashion_styles = []
    for category in style_categories.keys():
        fashion_styles.extend(style_categories[category])

    text_inputs = processor(text=fashion_styles, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        style_features = model.get_text_features(**text_inputs)

    print("✅ 読み込み完了！")
    return device, processor, model, fashion_styles, style_categories, style_features


# グローバルなリソースを一度だけ読み込む
device, processor, model, fashion_styles, style_categories, style_features = load_resources()

# --- 2. メイン処理のためのヘルパー関数 ---
def calculate_centroid_vector(uploaded_images, weights):
    """
    アップロードされた画像の重み付け平均（重心）ベクトルを計算します。
    
    Args:
        uploaded_images (list): アップロードされたPIL画像のリスト
        weights (list): 各画像の重要度を示す重みのリスト
        
    Returns:
        torch.Tensor: 計算された重心ベクトル
    """
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


# def display_style_analysis(query_features_centroid):
#     """
#     重心ベクトルに基づいて、スタイルの系統やカラーの分析結果を表示します。
    
#     Args:
#         query_features_centroid (torch.Tensor): 計算された重心ベクトル
#     """
#     st.header("Analysis Results (Style, Color, etc.)")
#     st.write("We analyzed the attributes that describe your uploaded outfits.")

#     for category_name, attributes in style_categories.items():
#         st.subheader(category_name)

#         labels = []
#         scores = []

#         # calculate similarity scores
#         for attribute in attributes:
#             try:
#                 attribute_index = fashion_styles.index(attribute)
#                 similarity_score = F.cosine_similarity(
#                     query_features_centroid,
#                     style_features[attribute_index].unsqueeze(0)
#                 ).item()
#                 st.write(f"**{attribute}**")
#                 st.progress(similarity_score)
#             except ValueError:
#                 continue

#         # create radar chart
#         if len(scores) > 0:
#             # radar chart は閉じる必要がある
#             labels.append(labels[0])
#             scores.append(scores[0])

#             fig = go.Figure()

#             fig.add_trace(go.Scatterpolar(
#                 r=scores,
#                 theta=labels,
#                 fill='toself'
#             ))

#             fig.update_layout(
#                 polar=dict(
#                     radialaxis=dict(
#                         visible=True,
#                         range=[0, 1]  # cosine similarity 想定
#                     )
#                 ),
#                 showlegend=False,
#                 margin=dict(l=20, r=20, t=20, b=20)
#             )

#             st.plotly_chart(fig, use_container_width=True)

def display_style_analysis(query_features_centroid):
    """
    重心ベクトルに基づいて、スタイルの系統やカラーの分析結果を表示します。
    
    Args:
        query_features_centroid (torch.Tensor): 計算された重心ベクトル
    """
    st.header("Analysis Results (Style, Color, etc.)")
    st.write("We analyzed the attributes that describe your uploaded outfits.")

    for category_name, attributes in style_categories.items():
        st.subheader(category_name)
        for attribute in attributes:
            try:
                attribute_index = fashion_styles.index(attribute)
                similarity_score = F.cosine_similarity(
                    query_features_centroid,
                    style_features[attribute_index].unsqueeze(0)
                ).item()
                st.write(f"**{attribute}**")
                st.progress(similarity_score)
            except ValueError:
                continue

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
            with st.spinner("Analyzing..."):
                # 重心ベクトルを計算
                query_features_centroid = calculate_centroid_vector(query_images, weights)

                # 計算が成功した場合にのみ次の処理を実行
                if query_features_centroid is not None:
                    # 各分析結果を表示
                    st.markdown("---")
                    display_style_analysis(query_features_centroid)


# アプリケーションの開始点
if __name__ == "__main__":
    main()
