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
    fashion_styles = [
        "casual style", "formal style", "vintage style", "modern style", 
        "minimalist style", "streetwear style", "bohemian style", "sporty style", 
        "elegant style", "vivid color", "monochrome color", "patterned", 
        "striped", "solid color", "oversized fit", "slim fit"
    ]
    text_inputs = processor(text=fashion_styles, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        style_features = model.get_text_features(**text_inputs)

    print("✅ 読み込み完了！")
    return device, processor, model, feature_tensor, all_images, fashion_styles, style_features

device, processor, model, feature_tensor, all_images, fashion_styles, style_features = load_resources()


# --- 2. ページのUI設定 ---
st.title("ファッション・イメージ分析アプリ")
st.write("複数の画像をアップロードすると、あなたの好みを分析し、それに合ったアイテムを提案します。")
uploaded_files = st.file_uploader("画像を複数アップロードしてください...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)


# --- 3. メイン処理：ベクトル計算と提案 ---
if uploaded_files:
    st.markdown("---")
    st.subheader("アップロードされた画像と重み付け")
    query_images = []
    weights = []

    # アップロード画像を横並びで表示し、重み付けスライダーを配置
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
        
        # --- 提案：テキスト提案 ---
        st.markdown("---")
        st.header("あなたの美的感覚を分析しました")
        st.write("アップロードされた画像から、あなたの好みを最も強く表すスタイルを分析しました。")

        similarities_to_styles = F.cosine_similarity(query_features_centroid, style_features)
        top_similarities, top_indices = torch.topk(similarities_to_styles, k=5)

        for i in range(5):
            style_name = fashion_styles[top_indices[i]]
            similarity_score = top_similarities[i].item()
            st.write(f"**{i+1}. {style_name}** (類似度: {similarity_score:.4f})")

        # --- 提案：画像提案 ---
        st.markdown("---")
        st.header("あなたにおすすめのアイテム")
        similarities_to_images = F.cosine_similarity(query_features_centroid, feature_tensor)
        best_match_index = torch.argmax(similarities_to_images).item()
        
        result_image = all_images[best_match_index]
        st.write("検索が完了しました！")
        st.image(result_image, caption=f"最も似ている画像 (類似度: {similarities_to_images.max():.4f})", use_column_width=True)