#convert to api
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
import tempfile
import shutil
from pathlib import Path
from paddleocr import PaddleOCRVL
import io

app = FastAPI(title="PaddleOCR Document Processing API")

# Initialize pipeline (reuse across requests for efficiency)
pipeline = PaddleOCRVL(
    vl_rec_backend="vllm-server", 
    vl_rec_server_url="http://34.48.55.215:8000/v1"
)

@app.post("/process-document")
async def process_document(file: UploadFile = File(...)):
    """
    Process uploaded document and return markdown file directly
    
    Args:
        file: Uploaded PDF or image file
    
    Returns:
        Markdown file with extracted text
    """
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save uploaded file
        input_file = temp_path / file.filename
        with open(input_file, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        try:
            # Process document
            print(f"Starting to process: {file.filename}")
            output = pipeline.predict(input=str(input_file))
            print(f"Processed {len(output)} pages")
            
            # Extract markdown
            markdown_list = []
            markdown_images = []
            
            for i, res in enumerate(output, 1):
                print(f"Processing page {i}/{len(output)}")
                md_info = res.markdown
                markdown_list.append(md_info)
                markdown_images.append(md_info.get("markdown_images", {}))
            
            # Concatenate markdown
            markdown_texts = pipeline.concatenate_markdown_pages(markdown_list)
            
            # Save images to temp directory (they're embedded as base64 in markdown)
            for item in markdown_images:
                if item:
                    for path, image in item.items():
                        file_path = temp_path / path
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        image.save(file_path)
            
            # Prepare output filename
            output_stem = Path(file.filename).stem
            md_filename = f"{output_stem}.md"
            
            # Return markdown file directly
            return Response(
                content=markdown_texts,
                media_type="text/markdown",
                headers={
                    "Content-Disposition": f"attachment; filename={md_filename}"
                }
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/process-document-json")
async def process_document_json(file: UploadFile = File(...)):
    """
    Alternative endpoint that returns JSON with markdown and metadata
    
    Args:
        file: Uploaded PDF or image file
    
    Returns:
        JSON object with markdown text and metadata
    """
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_file = temp_path / file.filename
        
        with open(input_file, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        try:
            output = pipeline.predict(input=str(input_file))
            
            markdown_list = []
            markdown_images = []
            
            for res in output:
                md_info = res.markdown
                markdown_list.append(md_info)
                markdown_images.append(md_info.get("markdown_images", {}))
            
            markdown_texts = pipeline.concatenate_markdown_pages(markdown_list)
            
            # Count images
            image_count = sum(len(item) for item in markdown_images if item)
            
            return {
                "filename": Path(file.filename).stem,
                "markdown": markdown_texts,
                "page_count": len(output),
                "image_count": image_count
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "PaddleOCR Processing API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

#curl -X POST "http://172.17.17.105:8080/process-document" -F "file=@d:/temp/LEXICON Employee-Handbook.pdf" --output result.md