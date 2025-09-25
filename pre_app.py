import torch
from transformers import AutoProcessor, AutoModel
from datasets import load_dataset
from tqdm import tqdm
import pickle
import os

def load_resources(model_name: str = "openai/clip-vit-base-patch32", 
                dataset_name: str = "ashraq/fashion-product-images-small"):
    """
    モデルとデータセットを読み込みます。

    Args:
        model_name (str): 使用するモデルの名前。
        dataset_name (str): 使用するデータセットの名前。

    Returns:
        tuple: (デバイス, プロセッサー, モデル, データセット)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    # モデルとプロセッサーを読み込み
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    print("✅ Model and processor loaded successfully!")

    # データセットをストリーミングモードで読み込み
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    print("✅ Dataset loaded successfully!")
    
    return device, processor, model, dataset

def process_and_save_data(device, processor, model, dataset, num_images: int = 10000):
    """
    データセットの画像を処理し、特徴量ベクトルと画像をファイルに保存します。

    Args:
        device (torch.device): 処理に使用するデバイス。
        processor: モデルのプロセッサー。
        model: 特徴量抽出に使用するモデル。
        dataset: 処理するデータセット。
        num_images (int): 処理する画像の数。
    """
    print(f"Starting image vectorization for {num_images} images...")

    all_image_features = []
    all_images = []

    # 指定された数の画像を処理
    for example in tqdm(dataset.take(num_images), total=num_images):
        image = example["image"]
        if image is None:
            continue
        
        image = image.convert("RGB")
        
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            
        all_image_features.append(image_features.cpu())
        all_images.append(image)

    # 計算したベクトルを1つのテンソルにまとめる
    feature_tensor = torch.cat(all_image_features)
    print(f"\n✅ Vectorized {len(all_image_features)} images.")
    print("Feature tensor shape:", feature_tensor.shape)

    # 計算結果をファイルに保存
    torch.save(feature_tensor, "features.pt")
    with open("images.pkl", "wb") as f:
        pickle.dump(all_images, f)

    print("✅ Features and images saved to files!")

def main():
    """
    スクリプトのメイン関数。リソースを読み込み、データを処理し、保存します。
    """
    device, processor, model, dataset = load_resources()
    process_and_save_data(device, processor, model, dataset)
    
if __name__ == "__main__":
    main()

