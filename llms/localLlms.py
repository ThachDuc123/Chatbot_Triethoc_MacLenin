import requests
import re
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from llms.onnx import ONNXModel
import json
import os

try:
    from huggingface_hub.errors import GatedRepoError
except Exception:
    GatedRepoError = None
class LocalLLMs:
    def __init__(self, engine: str, model_version: str, base_url: str = None, **kwargs):
        """ Initialize the LocalLLMs class 
            Args:
            engine (str): "ollama" or "vllm".
            model_version (str): name of model ("llama3","meta-llama/Llama-2-7b-chat-hf").
            base_url (str, optional): BASE URL of server engine.
        """
        self.engine = engine
        self.model_version = model_version
        self.client = None
        self.max_tokens = kwargs.get('max_tokens', 4096)  # Default max tokens
        # Long answers on slower machines can exceed 120s. Allow an env override.
        try:
            self.timeout_seconds = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))
        except Exception:
            self.timeout_seconds = 300
        if engine == "ollama":
            self.base_url = base_url 
            model_id = kwargs.get("model_name") or model_version
            # Keep self.model_version consistent with the actual model id we'll call.
            self.model_version = model_id
            self._initialize_ollama_model(model_id)
        elif engine == "vllm":
            self.base_url = base_url
            self._initialize_vllm_model(model_version)
        elif engine == "onnx":
            local_dir = kwargs.get('local_dir', './onnx_models')
            self.onnx_model = ONNXModel(model_version, local_dir)
            self.client = self.onnx_model
        elif engine == "huggingface":
            # Backwards-compatible:
            # - Historically we used `model_version` as the HF repo id.
            # - In `serve.py`, users expect `--model_name` to be the actual model identifier.
            # Prefer explicit `model_name` when provided.
            model_id = kwargs.get("model_name") or model_version
            self._initialize_huggingface_model(model_id)
        else:
            raise ValueError(f"Unsupported engine: {engine}")

    def _initialize_ollama_model(self, model_version: str):
        """Pull the specified model from the Ollama server."""
        try:
            # Normalize base_url so users can pass either:
            # - http://localhost:11434
            # - http://localhost:11434/v1
            # For Ollama's native API we want the root without /v1.
            if self.base_url and self.base_url.rstrip("/").endswith("/v1"):
                self.base_url = self.base_url.rstrip("/")[:-3]

            response = requests.get(self.base_url, timeout=5)
            response.raise_for_status()
            print("Kết nối đến máy chủ Ollama thành công.")
            self.client = requests.Session()
            self._pull_ollama_model(model_version)
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Không thể kết nối đến máy chủ Ollama tại {self.base_url}. Vui lòng đảm bảo Ollama đang chạy. Lỗi: {e}")

    def _pull_ollama_model(self, model_version: str):
        """Pull model from Ollama if not exist."""
        try:
            # 1. Check if the model already exists
            response = self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            model_exists = any(model_version in m["name"] for m in models)

            # 2. If the model does not exist, pull it
            if not model_exists:
                print(f"Model '{model_version}' chưa tồn tại. Bắt đầu tải...")
                pull_data = {"name": model_version}
                pull_response = self.client.post(f"{self.base_url}/api/pull", json=pull_data)
                pull_response.raise_for_status()
                print(f"Tải model '{model_version}' thành công.")
            else:
                print(f"Model '{model_version}' đã có sẵn.")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Lỗi khi giao tiếp với API của Ollama. Lỗi: {e}")

    def _initialize_vllm_model(self, model_version: str):
        """Initialize the vLLM model with the specified name and parameters."""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=10)
            response.raise_for_status()
            models = response.json().get("data", [])
            matched_model = next((m for m in models if m["id"] == self.model_version), None)

            if matched_model:
                self.max_tokens = matched_model.get("max_model_len", 4096)
                print(f"Model '{self.model_version}' đã được tìm thấy với max_tokens: {self.max_tokens}.")
            else:
                print(f"Không tìm thấy model '{self.model_version}' trong danh sách model của vLLM. Dùng giá trị mặc định 4096.")

            print("Kết nối đến vLLM server thành công.")
            self.client = requests.Session()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Không thể kết nối đến vLLM tại {self.base_url}. Lỗi: {e}")
    
    def _initialize_huggingface_model(self, model_version: str):
        """Initialize the Huggingface model with the specified name and parameters"""

        # If device_map='auto' is used, Accelerate may offload some layers to CPU/disk.
        # In that case, calling .to(device) will crash with:
        # "You can't move a model that has some modules offloaded to cpu or disk."
        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_version,
                trust_remote_code=True,
            )
        except Exception as e:
            # Common Windows/offline failure: gated repo that requires HF login.
            msg = str(e)
            is_gated = (
                (GatedRepoError is not None and isinstance(e, GatedRepoError))
                or "gated repo" in msg.lower()
                or "401" in msg
                or "unauthorized" in msg.lower()
            )
            if is_gated:
                raise RuntimeError(
                    "Model HuggingFace này đang bị giới hạn quyền truy cập (gated/private), nên bạn bị lỗi 401 Unauthorized.\n\n"
                    f"Model: {model_version}\n\n"
                    "Cách sửa nhanh:\n"
                    "1) Dùng model public (không bị gated), ví dụ: Qwen/Qwen2.5-3B-Instruct hoặc TinyLlama/TinyLlama-1.1B-Chat-v1.0\n"
                    "2) Hoặc đăng nhập HuggingFace: cài huggingface-cli và chạy 'huggingface-cli login' (cần token có quyền).\n"
                    "3) Hoặc dùng chế độ offline Ollama (khuyến nghị trong HUONG_DAN_OFFLINE.md).\n\n"
                    f"Chi tiết lỗi: {msg}"
                ) from e
            raise
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.client = AutoModelForCausalLM.from_pretrained(
            model_version,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code = True,
            low_cpu_mem_usage=True,
        )

        # Only move the model manually when we did NOT ask Accelerate to dispatch it.
        # (We currently always use device_map='auto', so this is a no-op and safe.)
        if getattr(self.client, "hf_device_map", None) is None:
            self.client = self.client.to(device)
        self.client.eval()

    def remove_think_blocks(self,text):
        """Remove <think> blocks and their content from text"""
        # Pattern to match <think>...</think> blocks (including multiline)
        pattern = r'<think>.*?</think>'
        # Remove the think blocks using re.sub with DOTALL flag for multiline matching
        cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
        # Clean up any extra whitespace that might be left
        cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text).strip()
        return cleaned_text
    
    def generate_content(self, prompt: List[Dict[str,str]]) -> str:
        """Generate content using the local LLM based on the provided prompt.
            input: prompt (str): The prompt to generate content for.
            output: str: The generated content.
        """
        if not self.client:
            raise RuntimeError("Client chưa được khởi tạo. Vui lòng kiểm tra lại cấu hình.")

        print(f"Đang tạo nội dung với engine '{self.engine}' và model '{self.model_version}'...")

        try:
            if self.engine == 'ollama':
                payload = {
                    "model": self.model_version,
                    "messages": prompt,
                    "stream": False,
                    # Cap generation length for speed.
                    "options": {
                        "num_predict": int(self.max_tokens) if self.max_tokens else 256,
                    },
                }
                # Try Ollama native endpoint first.
                response = self.client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.timeout_seconds,
                )

                # Some setups expose only the OpenAI-compatible endpoint.
                if response.status_code == 404:
                    oa_payload = {
                        "model": self.model_version,
                        "messages": prompt,
                        "stream": False,
                    }
                    response = self.client.post(
                        f"{self.base_url}/v1/chat/completions",
                        headers={"Content-Type": "application/json"},
                        json=oa_payload,
                        timeout=self.timeout_seconds,
                    )

                response.raise_for_status()

                data = response.json()
                # native: {message:{content:...}}
                if isinstance(data, dict) and "message" in data:
                    text = data["message"]["content"].strip()
                    return self.remove_think_blocks(text)
                # openai-compatible: {choices:[{message:{content:...}}]}
                if isinstance(data, dict) and "choices" in data:
                    text = data["choices"][0]["message"]["content"].strip()
                    return self.remove_think_blocks(text)

                raise RuntimeError(
                    f"Ollama response format not recognized: keys={list(data.keys()) if isinstance(data, dict) else type(data)}"
                )

            elif self.engine == 'vllm':
                payload = {
                    "model": self.model_version,
                    "messages": prompt,
                    # "max_tokens": self.max_tokens,
                    # "temperature": 0.7
                }
                response = self.client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json=payload
                )
                response.raise_for_status()
                response_data = response.json()["choices"][0]["message"]["content"].strip()
                return self.remove_think_blocks(response_data)
            elif self.engine == 'huggingface':
                # messages = [
                #     {"role": "user", "content": prompt}
                # ]
                import torch
                messages = prompt
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False # thinking mode unabled
                )

                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.client.device)

                # conduct text completion
                with torch.no_grad():
                    generated_ids = self.client.generate(
                        **model_inputs,
                        max_new_tokens=self.max_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True
                    )
                output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

                response_data = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                return self.remove_think_blocks(response_data)
            elif self.engine == "onnx":
                # TODO: Will find the prompt template of each model
                if isinstance(prompt, list):
                    prompt_text = ""
                    for msg in prompt:
                        if msg["role"] == "system":
                            prompt_text += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
                        elif msg["role"] == "user":
                            prompt_text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
                        elif msg["role"] == "assistant":
                            prompt_text += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
                    prompt_text += "<|im_start|>assistant\n"  # Model is expected to complete from here
                else:
                    prompt_text = prompt  # Assume already formatted

                output = self.onnx_model.generate(prompt_text)
                return self.remove_think_blocks(output)

            raise ValueError(f"Unsupported engine: {self.engine}")

        except Exception as e:
            print(f"Đã xảy ra lỗi trong quá trình tạo nội dung: {e}")
            raise

    def stream_content(self, prompt: List[Dict[str, str]]):
        """Yield generated text chunks for engines that support streaming.

        Currently implemented for Ollama only.
        """
        if not self.client:
            raise RuntimeError("Client chưa được khởi tạo. Vui lòng kiểm tra lại cấu hình.")

        if self.engine != "ollama":
            raise NotImplementedError("Streaming is only supported for Ollama in this project")

        payload = {
            "model": self.model_version,
            "messages": prompt,
            "stream": True,
            "options": {
                "num_predict": int(self.max_tokens) if self.max_tokens else 256,
            },
        }

        # Native streaming endpoint.
        r = self.client.post(
            f"{self.base_url}/api/chat",
            json=payload,
            stream=True,
            timeout=self.timeout_seconds,
        )
        if r.status_code == 404:
            # OpenAI-compatible streaming (not always enabled in Ollama, but try).
            oa_payload = {
                "model": self.model_version,
                "messages": prompt,
                "stream": True,
            }
            r = self.client.post(
                f"{self.base_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=oa_payload,
                stream=True,
                timeout=self.timeout_seconds,
            )

        r.raise_for_status()

        # Ollama native: newline-delimited JSON objects
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            # native format
            if isinstance(obj, dict) and "message" in obj:
                content = obj.get("message", {}).get("content") or ""
                if content:
                    yield self.remove_think_blocks(content)
                if obj.get("done") is True:
                    break
                continue

            # openai format
            if isinstance(obj, dict) and "choices" in obj:
                delta = obj["choices"][0].get("delta", {})
                content = delta.get("content") or ""
                if content:
                    yield self.remove_think_blocks(content)
                if obj["choices"][0].get("finish_reason"):
                    break
        