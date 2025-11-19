from pathlib import Path
from paddleocr import PaddleOCRVL


input_file = r"D:\temp\Technical Delivery Manager - Cards .pdf"
output_path = Path(".")

pipeline = PaddleOCRVL(vl_rec_backend="vllm-server", vl_rec_server_url="http://34.48.55.215:8000/v1")
#pipeline = PaddleOCRVL()

output = pipeline.predict(input=input_file)


markdown_list = []
markdown_images = []

for res in output:
    md_info = res.markdown
    markdown_list.append(md_info)
    markdown_images.append(md_info.get("markdown_images", {}))

markdown_texts = pipeline.concatenate_markdown_pages(markdown_list)

mkd_file_path = output_path / f"{Path(input_file).stem}.md"
mkd_file_path.parent.mkdir(parents=True, exist_ok=True)

with open(mkd_file_path, "w", encoding="utf-8") as f:
    f.write(markdown_texts)

for item in markdown_images:
    if item:
        for path, image in item.items():
            file_path = output_path / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(file_path)



#https://ernie.baidu.com/blog/posts/paddleocr-vl/
#https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL.html#22-python-script-integration
#https://pub.towardsai.net/docling-an-opensource-python-library-for-pdf-parsing-ocr-support-rag-ibm-research-fe6177235329?sk=7d7b1aa05feb1e0dc907ca6ceb60323c
#https://github.com/PaddlePaddle/PaddleOCR/issues/16711
#vllm serve PaddlePaddle/PaddleOCR-VL --served-model-name PaddleOCR-VL-0.9B --trust-remote-code --max-num-batched-tokens 131072 --max-model-len 131072 --no-enable-prefix-caching  --host 0.0.0.0  --mm-processor-cache-gb 0