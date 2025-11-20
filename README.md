# RAG Pipeline with PaddleOCR-VL, vLLM, and n8n

This project is a Proof of Concept (PoC) for an end-to-end RAG (Retrieval-Augmented Generation) pipeline. It uses the powerful PaddleOCR-VL model for Optical Character Recognition (OCR), served via vLLM for high-performance inference. The entire process, from document ingestion to the final chat agent, is orchestrated using n8n workflows.

The system is designed to take PDF documents, extract their text and layout information into Markdown, store this structured data in a vector database, and then allow a user to ask questions about the documents through an AI agent.

## Project Components

1.  **VLLM PaddleOCR-VL Server**: A dedicated Python server running on a GPU-enabled machine to host the `PaddlePaddle/PaddleOCR-VL` model. This provides the core OCR capabilities.
2.  **FastAPI OCR Service (`paddleOcr-vl-v5.py`)**: A Python-based API that acts as a bridge between n8n and the vLLM server. It accepts PDF uploads, splits them page by page, sends them to the vLLM server for processing, and returns the final concatenated Markdown.
3.  **n8n PDF Ingestion Workflow (`n8n-PDF2MD2Vector.json`)**: An n8n workflow that provides a UI for uploading PDFs. It calls the FastAPI service to perform OCR, and then chunks, embeds (using OpenAI), and stores the resulting Markdown into a PGVector database. It also includes a record manager to avoid reprocessing unchanged files.
4.  **n8n RAG Agent Workflow (`n8n-AI Agent with Vectors.json`)**: The user-facing component. This n8n workflow provides a chat interface powered by an AI agent (e.g., OpenAI GPT models). When a user asks a question, the agent retrieves relevant context from the PGVector database to provide accurate, source-based answers.

## Architecture Flow

Here is a high-level overview of the data flow for both ingestion and retrieval:

**Document Ingestion:**
`User Uploads PDF` -> `n8n Ingestion Workflow` -> `FastAPI OCR Service` -> `vLLM PaddleOCR-VL Server` -> `Markdown Output` -> `n8n (Embed & Store)` -> `PGVector Database`

**AI Agent Chat:**
`User Asks Question` -> `n8n AI Agent` -> `PGVector (Similarity Search)` -> `Retrieved Context + Question` -> `OpenAI LLM` -> `Generated Answer` -> `User`

## Setup and Installation

### Prerequisites

*   An **n8n** instance (Cloud or self-hosted).
*   A **PostgreSQL** database with the **pgvector** extension enabled.
*   An **OpenAI API Key**.
*   A dedicated server with a powerful **NVIDIA GPU** (e.g., A100, H100) with **at least 10GB VRAM** for running the vLLM server. This server can also be used to host other large language models (e.g., embedding or chat models) if sufficient VRAM is available.
*   **CUDA** drivers and toolkit properly installed on the GPU server. (e.g., for cloud GPU setup, [brev.nvidia.com](https://brev.nvidia.com/) is a useful resource for pre-configured environments).
*   **Python 3.10+** and `pip` for the FastAPI service.

### Step 1: Set up the vLLM PaddleOCR-VL Server

On your dedicated GPU server, install vLLM and its dependencies.

```bash
pip install vllm
```

Launch the vLLM server with the `PaddlePaddle/PaddleOCR-VL` model. It's critical to use the `--served-model-name` argument because the Python script specifically looks for `PaddleOCR-VL-0.9B`.

```bash
vllm serve PaddlePaddle/PaddleOCR-VL \
  --served-model-name PaddleOCR-VL-0.9B \
  --trust-remote-code \
  --max-num-batched-tokens 131072 \
  --max-model-len 131072 \
  --no-enable-prefix-caching \
  --host 0.0.0.0 \
  --mm-processor-cache-gb 0
```

Take note of the server's IP address and port (default is `8000`).

### Step 2: Set up the FastAPI OCR Service

1.  Clone this repository or download the `paddleOcr-vl-v5.py` and `requirements.txt` files.
2.  Install the required Python packages:
    ```bash
pip install -r requirements.txt
    ```
3.  **Update the vLLM server URL**: Open `paddleOcr-vl-v5.py` and modify the `vl_rec_server_url` to point to the IP address and port of your vLLM server from Step 1.

    ```python
    # paddleOcr-vl-v5.py

    # ... around line 26
    pipeline = PaddleOCRVL(
        vl_rec_backend="vllm-server", 
        vl_rec_server_url="http://<YOUR_VLLM_SERVER_IP>:8000/v1" # <--- CHANGE THIS
    )
    # ... 
    ```

4.  Run the FastAPI server. It will listen on port `8080` by default.
    ```bash
uvicorn paddleOcr-vl-v5:app --host 0.0.0.0 --port 8080
    ```

### Step 3: Configure n8n Workflows

1.  **Import Workflows**: Import both `n8n-PDF2MD2Vector.json` and `n8n-AI Agent with Vectors.json` into your n8n instance.

2.  **Configure `PDF2MD2Vector` Workflow**:
    *   **Update HTTP Request URL**: Find the **"Submit to Python API"** node and change the URL to the IP address and port of your FastAPI OCR service (from Step 2).
    *   **Set up Credentials**:
        *   **Postgres PGVector Store**: Configure your Postgres (pgvector) database credentials.
        *   **Embeddings OpenAI**: Configure your OpenAI API credentials.
        *   **Try Match File in Record Manager**: Configure the same Postgres credentials.
        *   **Add Filename to Record Manager**: Configure the same Postgres credentials.
        *   **File Changed remove old file and RAG**: Configure the same Postgres credentials.

3.  **Configure `AI Agent with Vectors` Workflow**:
    *   **Set up Credentials**:
        *   **Postgres Chat Memory**: Configure your Postgres credentials for storing chat history.
        *   **OpenAI Chat Model**: Configure your OpenAI credentials.
        *   **Get Data from RAG**: Configure your Postgres (pgvector) credentials.
        *   **Embeddings OpenAI**: Configure the same OpenAI credentials.

4.  **Activate Workflows**: Enable both workflows in the n8n UI.

## Usage

1.  **Ingest a Document**:
    *   Go to the `PDF2MD2Vector` workflow in n8n.
    *   Click the "View" button on the **"Request to upload only 1 file"** (Form Trigger) node to get the form URL.
    *   Open the URL, upload a PDF file, and submit. The workflow will start processing the file. You can monitor its progress in the n8n "Executions" tab.

2.  **Chat with your Document**:
    *   Go to the `AI Agent with Vectors` workflow.
    *   The **"Incoming Chat"** node provides a chat interface. Open it and start asking questions about the content of the PDF you just uploaded.

## Important Notes

*   **IP Addresses are Placeholders**: The IP addresses in the provided files (`34.48.5.42`, `172.17.17.105`) are examples. You **must** replace them with the actual IP addresses of your running services.
*   **Credentials**: This project requires multiple credentials (Postgres, OpenAI). Ensure they are configured correctly in all relevant n8n nodes.
*   **Performance**: The performance of the OCR process heavily depends on the GPU running the vLLM server. The page-by-page processing in the FastAPI script is a deliberate choice to handle large documents and provide progress feedback, but it may be slower than a bulk processing approach.
*   **Cost**: This pipeline uses the OpenAI API for embeddings and chat completions, which will incur costs. Monitor your usage accordingly.
