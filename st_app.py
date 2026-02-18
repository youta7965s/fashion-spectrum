import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
import pickle

# --- 1. モデルとデータの読み込み ---
# Streamlitのキャッシュ機能を使って、アプリケーションの実行中に一度だけリソースを読み込みます。
# これにより、ユーザーがUIを操作するたびに再読み込みされるのを防ぎ、高速化します。
@st.cache_resource
def load_resources():
    """
    アプリケーションに必要なモデル、データ、および特徴量を読み込みます。
    
    Returns:
        tuple: 必要なリソース（デバイス、プロセッサー、モデル、特徴量、画像、スタイルデータなど）
    """
    print("✅ リソースを読み込み中...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "openai/clip-vit-base-patch32"
    
    # Hugging FaceからCLIPモデルとプロセッサーを読み込み
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    # 事前に作成したデータベースの画像特徴量と画像データを読み込み
    # このファイルはアプリケーション実行環境に存在する必要があります
    try:
        feature_tensor = torch.load("features.pt").to(device)
        with open("images.pkl", "rb") as f:
            all_images = pickle.load(f)
    except FileNotFoundError:
        st.error("必要なデータベースファイル (features.pt, images.pkl) が見つかりません。")
        st.stop()
    
    # スタイル提案用のテキストを定義し、ベクトル化
    style_categories = {
        "系統": ["streetwear", "vintage", "modern", "sporty", "elegant", "preppy", "minimalist", "punk", "gothic", "hippie", "grunge"],
        "カラー": ["red", "blue", "green", "yellow", "black", "white", "pink", "purple", "orange", "brown", "gray"],
        "トーン": ["vivid color", "monochrome color", "pastel color", "dark tone"],
        "シルエット": ["A-line silhouette", "I-line silhouette", "H-line silhouette", "O-line silhouette"],
        "サイズ": ["oversized fit", "slim fit"]
    }
    fashion_styles = []
    for category in style_categories.keys():
        fashion_styles.extend(style_categories[category])

    text_inputs = processor(text=fashion_styles, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        style_features = model.get_text_features(**text_inputs)

    # カジュアル・フォーマル分析のための基準ベクトルを計算
    formal_text = "a very formal outfit"
    formal_input = processor(text=formal_text, return_tensors="pt").to(device)
    with torch.no_grad():
        formal_features = model.get_text_features(**formal_input)

    
    print("✅ 読み込み完了！")
    return device, processor, model, feature_tensor, all_images, fashion_styles, style_categories, style_features, formal_features

# グローバルなリソースを一度だけ読み込む
device, processor, model, feature_tensor, all_images, fashion_styles, style_categories, style_features, formal_features = load_resources()

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
        st.warning("重みがすべて0です。少なくとも1つの画像の重みを0より大きい値に設定してください。")
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

def display_formality_analysis(query_features_centroid):
    """
    重心ベクトルに基づいて、フォーマル度とカジュアル度を表示します。
    
    Args:
        query_features_centroid (torch.Tensor): 計算された重心ベクトル
    """
    st.header("分析結果（フォーマル or カジュアル）")
    
    # フォーマルスコアを計算
    formal_score = F.cosine_similarity(query_features_centroid, formal_features).item()

    # スコアの範囲を正規化（必要に応じてformal_min, formal_maxを調整してください）
    formal_min, formal_max = 0.20, 0.28 
    normalized_formal_score = (formal_score - formal_min) / (formal_max - formal_min)
    normalized_formal_score = max(0.0, min(1.0, normalized_formal_score))
    
    # カジュアルスコアは「フォーマルではない」として定義
    casual_score = 1 - normalized_formal_score

    st.write(f"**フォーマル度 ({normalized_formal_score:.4f})**")
    st.progress(normalized_formal_score)
    st.write(f"**フォーマル度 ({casual_score:.4f})**")
    st.progress(casual_score)

def display_style_analysis(query_features_centroid):
    """
    重心ベクトルに基づいて、スタイルの系統やカラーの分析結果を表示します。
    
    Args:
        query_features_centroid (torch.Tensor): 計算された重心ベクトル
    """
    st.header("分析結果（系統、カラーなど）")
    st.write("アップロードされた画像から、あなたのスタイリングを構成する要素を分析しました。")

    for category_name, attributes in style_categories.items():
        st.subheader(category_name)
        for attribute in attributes:
            try:
                attribute_index = fashion_styles.index(attribute)
                similarity_score = F.cosine_similarity(query_features_centroid, style_features[attribute_index].unsqueeze(0)).item()
                st.write(f"**{attribute}**")
                st.progress(similarity_score)
            except ValueError:
                continue

def display_image_recommendation(query_features_centroid):
    """
    重心ベクトルに最も類似した画像をデータベースから見つけて表示します。
    
    Args:
        query_features_centroid (torch.Tensor): 計算された重心ベクトル
    """
    st.header("あなたにおすすめのアイテム")
    
    # データベース内の画像ベクトルと重心ベクトルの類似度を計算
    similarities_to_images = F.cosine_similarity(query_features_centroid, feature_tensor)
    
    # 最も類似度の高い画像を見つける
    best_match_index = torch.argmax(similarities_to_images).item()
    result_image = all_images[best_match_index]
    
    st.write("検索が完了しました！")
    st.image(result_image, caption=f"最も似ている画像 (類似度: {similarities_to_images.max():.4f})", use_column_width=True)

# --- 3. Streamlit アプリケーション本体 ---
def main():
    """
    Streamlitアプリケーションのメイン関数。UIの構築と処理の流れを定義します。
    """
    st.title("ファッションイメージ分析アプリ")
    st.write("スタイリングやアイテム画像を、系統・色・形といった属性に分解します。")
    
    uploaded_files = st.file_uploader(
        "画像をアップロードしてください...", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.markdown("---")
        st.subheader("アップロードされた画像と重み付け")
        query_images = []
        weights = []

        # アップロード画像を表示し、重み付けスライダーを配置
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.image(image)
            with col2:
                # 各画像に固有のキーを持つスライダーを配置
                weight = st.slider(
                    "この画像の重要度", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=0.5, 
                    step=0.05, 
                    key=f"slider_{uploaded_file.name}"
                )
            query_images.append(image)
            weights.append(weight)

        # 分析実行ボタン
        st.markdown("---")
        if st.button("分析を実行"):
            with st.spinner("分析中..."):
                # 重心ベクトルを計算
                query_features_centroid = calculate_centroid_vector(query_images, weights)
                
                # 計算が成功した場合にのみ次の処理を実行
                if query_features_centroid is not None:
                    # 各分析結果を表示
                    st.markdown("---")
                    display_formality_analysis(query_features_centroid)
                    st.markdown("---")
                    display_style_analysis(query_features_centroid)
                    st.markdown("---")
                    display_image_recommendation(query_features_centroid)

# アプリケーションの開始点
if __name__ == "__main__":
    main()
