import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
import pickle

# --- 1. モデルとデータの読み込み ---
@st.cache_resource
def load_resources():
    print("✅ リソースを読み込み中...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "openai/clip-vit-base-patch32"
    
    # モデルとプロセッサーを読み込み
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    # データベースの画像とベクトルを読み込み
    feature_tensor = torch.load("features.pt").to(device)
    with open("images.pkl", "rb") as f:
        all_images = pickle.load(f)
        
    # スタイル提案用のテキストをベクトル化
    style_categories = {
        "系統": ["streetwear", "vintage", "modern", "bohemian", "sporty", "elegant", "chic", "preppy", "y2k", "minimalist", "classic", "punk", "gothic", "hippie", "grunge"],
        "カラー": ["red", "blue", "green", "yellow", "black", "white", "pink", "purple", "orange", "brown", "gray"],
        # "トーン": ["vivid color", "monochrome color", "pastel color", "earth tone"],
        # "サイズ": ["oversized fit", "slim fit"],
        # "シルエット": ["A-line silhouette", "I-line silhouette", "H-line silhouette", "O-line silhouette"]
    }
    
    fashion_styles = []
    for category in style_categories.keys():
        fashion_styles.extend(style_categories[category])

    text_inputs = processor(text=fashion_styles, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        style_features = model.get_text_features(**text_inputs)

    # カジュアル・フォーマル分析のための基準フォーマルベクトルを計算
    formal_text = "a very formal outfit"
    formal_input = processor(text=formal_text, return_tensors="pt").to(device)
    with torch.no_grad():
        formal_features = model.get_text_features(**formal_input)

    print("✅ 読み込み完了！")
    return device, processor, model, feature_tensor, all_images, fashion_styles, style_categories, style_features, formal_features

device, processor, model, feature_tensor, all_images, fashion_styles, style_categories, style_features, formal_features = load_resources()


# --- 2. ページのUI設定 ---
st.title("スタイリング分析アプリ")
st.write("複数の画像をアップロードすると、あなたの好みを分析し、それに合ったアイテムを提案します。")
uploaded_files = st.file_uploader("画像をアップロードしてください...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)


# --- 3. メイン処理：ベクトル計算と提案 ---
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

    # 重心ベクトルの計算
    if st.button("分析を実行"):
        weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
        if weights_tensor.sum() == 0:
            st.warning("重みがすべて0です。少なくとも1つの画像の重みを1以上に設定してください。")
            st.stop()
        
        all_query_features = []
        for image in query_images:
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            all_query_features.append(image_features)

        weighted_features = [feat * weight for feat, weight in zip(all_query_features, weights_tensor)]
        weighted_sum = torch.sum(torch.stack(weighted_features), dim=0)
        query_features_centroid = weighted_sum / weights_tensor.sum()


        # --- 提案：フォーマル度とカジュアル度の分析 ---
        st.markdown("---")
        st.header("分析結果（フォーマル or カジュアル）")
        
        # フォーマルスコアを直接計算
        formal_score = F.cosine_similarity(query_features_centroid, formal_features).item()

        # 様々な画像を入力してみて、フォーマル度の分布範囲を確認してください
        # 最小値（最もカジュアル）と最大値（最もフォーマル）を設定してください
        # 万が一機能しない場合、この値を調整してください
        formal_min, formal_max = 0.20, 0.28 # 目安の値

        # スコアの範囲を正規化
        normalized_formal_score = (formal_score - formal_min) / (formal_max - formal_min)
        normalized_formal_score = max(0.0, min(1.0, normalized_formal_score)) # 0-1の範囲にクリッピング
        
        # カジュアルスコアを「フォーマルではない」として定義
        casual_score = 1 - normalized_formal_score

        # st.write("**テスト用：正規化していないフォーマルスコア**")
        # st.progress(normalized_formal_score)

        # フォーマル度とカジュアル度を進捗バーで表示
        st.write(f"**フォーマル度 ({normalized_formal_score:.4f})**")
        st.progress(normalized_formal_score)
        st.write(f"**カジュアル度 ({casual_score:.4f})**")
        st.progress(casual_score)


        # --- 提案：テキスト提案 ---
        st.markdown("---")
        st.header("分析結果（系統、カラー、トーン...）")
        st.write("アップロードされた画像から、あなたのスタイリングを構成する要素を分析しました。")

        # 各カテゴリの分析結果を表示
        for category_name, attributes in style_categories.items():
            st.subheader(category_name)
            
            for attribute in attributes:
                # このテキストに対応するベクトルのインデックスを特定
                try:
                    # fashion_stylesリスト内でattributeのインデックスを検索
                    attribute_index = fashion_styles.index(attribute)
                except ValueError:
                    # テキストが見つからない場合の処理
                    continue
                
                # 類似度を計算
                similarity_score = F.cosine_similarity(query_features_centroid, style_features[attribute_index].unsqueeze(0)).item()

                # 結果を表示
                st.write(f"**{attribute}**")
                st.progress(similarity_score)
                st.write(f"類似度: {similarity_score:.4f}")

        st.markdown("---")


        # --- 提案：画像提案 ---
        st.markdown("---")
        st.header("あなたにおすすめのアイテム")
        similarities_to_images = F.cosine_similarity(query_features_centroid, feature_tensor)
        best_match_index = torch.argmax(similarities_to_images).item()
        
        result_image = all_images[best_match_index]
        st.write("検索が完了しました！")
        st.image(result_image, caption=f"最も似ている画像 (類似度: {similarities_to_images.max():.4f})", use_column_width=True)