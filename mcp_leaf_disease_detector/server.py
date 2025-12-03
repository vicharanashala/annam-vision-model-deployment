
# server.py — MCP wrapper for your FastAPI leaf disease backend
# import necessary libraries
from mcp.server.fastmcp import FastMCP
import httpx
import base64
import os

BACKEND_URL = "http://localhost:8001/predict"

mcp = FastMCP("mcp_server_leaf_disease_detector")


@mcp.tool()
def predict_leaf_disease(image_path: str) -> dict:
    """
    Send an image file to the FastAPI backend and return the prediction.
    """

    if not os.path.exists(image_path):
        return {"error": f"File not found: {image_path}"}

    with open(image_path, "rb") as f:
        file_data = f.read()

    # Send image to FastAPI backend
    files = {"file": ("image.jpg", file_data, "image/jpeg")}

    try:
        response = httpx.post(BACKEND_URL, files=files, timeout=60)
        response.raise_for_status()
    except Exception as e:
        return {"error": f"Backend request failed: {e}"}

    return response.json()


if __name__ == "__main__":
    mcp.run()
