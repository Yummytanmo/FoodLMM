from gradio_client import Client
from PIL import Image

def main():
    # 连接到运行中的 Gradio 服务
    client = Client("http://localhost:7618")
    
    # 准备输入数据
    image_path = "pizza.jpg"  # 替换为您的图片路径
    image = Image.open(image_path)
    
    # 设置生成参数
    temperature = 0.2
    top_p = 0.7
    max_output_tokens = 512
    
    # 准备提示文本
    prompt = "Can you identify a dessert item in this image that I could indulge in to satisfy my sweet tooth? Please output segmentation mask."
    
    # 调用生成接口
    result = client.predict(
        image,                  # 图片输入
        prompt,                 # 文本提示
        temperature,           # 温度参数
        top_p,                 # top-p 参数
        max_output_tokens,     # 最大输出 token 数
        api_name="/predict"    # API 端点名称
    )
    
    # 打印结果
    print("Generation Result:", result)

if __name__ == "__main__":
    main() 