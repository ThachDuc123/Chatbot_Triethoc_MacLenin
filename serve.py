import os

# ---- IMPORTANT (Windows stability) ----
# This project uses SentenceTransformer embeddings (PyTorch). Some versions of
# transformers/sentence-transformers try to auto-import TensorFlow/Keras, which
# can crash or hang on certain Windows setups. Disable TF backend explicitly.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

from flask import Flask, request, jsonify, Response, send_from_directory
from dotenv import load_dotenv
from flask_cors import CORS
from rag.core import RAG
from embeddings import SentenceTransformerEmbedding, EmbeddingConfig
from semantic_router import SemanticRouter, Route
from semantic_router.samples import productsSample, chitchatSample
import openai
from reflection import Reflection
from re_rank import Reranker
from llms.llms import LLMs
import argparse
import warnings
from insert_data import load_csv_to_chromadb
import re
import pandas as pd
import uuid
import time

# Load environment variables from .env file
load_dotenv()
# Optional: silence oneDNN numerical-difference warning (not an error).
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# Add custom exception
class URLNotFoundError(Exception):
    def __init__(self, name):
        self.name = name 
        super().__init__(f"Please make sure you have {name} in .env")

class APINotFoundError(Exception):
    def __init__(self, name):
        self.name = name 
        super().__init__(f"Please make sure you have {name} in .env")

class ValueNotFoundError(Exception):
    def __init__(self, name):
        self.name = name 
        super().__init__(f"Please make sure you have {name} in .env")

class DataNotFoundError(Exception):
    def __init__(self):
        super().__init__(f"Please make sure you have valid CSV file in folder data")

def main(args):

    # Only import Gemini SDK when needed (online gemini mode). This keeps offline
    # startup fast and avoids pulling extra deps like aiohttp.
    if args.mode == "online" and args.model_name == "gemini":
        import google.generativeai as genai  # noqa: F401

    # --- Semantic Router Setup --- #

    # define products route name
    PRODUCT_ROUTE_NAME = 'products' 
    CHITCHAT_ROUTE_NAME = 'chitchat'

    # Embedding model can be heavy to download on first run. Allow a fast/local-friendly
    # fallback via --embedding_backend fastembed to avoid huge HuggingFace downloads.
    if getattr(args, "embedding_backend", "sentence_transformers") == "fastembed":
        from embeddings.fastEmbed import FastEmbedding
        sentenceTransformerEmbedding = FastEmbedding(name=args.embedding_model)
    else:
        sentenceTransformerEmbedding = SentenceTransformerEmbedding(config=EmbeddingConfig(name=args.embedding_model))
    productRoute = Route(name=PRODUCT_ROUTE_NAME, samples=productsSample)
    chitchatRoute = Route(name=CHITCHAT_ROUTE_NAME, samples=chitchatSample)
    semanticRouter = SemanticRouter(sentenceTransformerEmbedding, routes=[productRoute, chitchatRoute])
    
    # --- End Semantic Router Setup --- #

    # --- Set up LLMs --- #

    if args.mode == "online" and args.model_name == "gemini":
        MODEL_API_KEY = os.getenv('GEMINI_API_KEY', None)  
        MODEL_BASE_URL = None

        if not MODEL_API_KEY:
            raise APINotFoundError('GEMINI_API_KEY')

    elif args.mode == "online" and args.model_name == "openai":
        MODEL_API_KEY = os.getenv('OPENAI_API_KEY')
        MODEL_BASE_URL = None

        if not MODEL_API_KEY:
            raise APINotFoundError('OPENAI_API_KEY')

    elif args.mode == "online" and args.model_name == "together":
        MODEL_API_KEY = os.getenv('TOGETHER_API_KEY', None)
        MODEL_BASE_URL = os.getenv("TOGETHER_BASE_URL", None)

        if not MODEL_API_KEY:
            raise APINotFoundError('TOGETHER_API_KEY')
        if not MODEL_BASE_URL:
            raise URLNotFoundError('TOGETHER_BASE_URL')

    elif args.mode == "offline" and args.model_engine == "ollama":
        MODEL_API_KEY = None
        MODEL_BASE_URL = os.getenv("OLLAMA_BASE_URL", None)

        # PowerShell line-wrapping can break --model_name (e.g. "qwen2.5\n:7b-instruct").
        # Allow a robust env var override.
        if getattr(args, "model_name", None):
            args.model_name = str(args.model_name).replace("\n", "").replace("\r", "").strip()
        env_model = os.getenv("OLLAMA_MODEL", "").strip()
        if env_model:
            args.model_name = env_model

        if not MODEL_BASE_URL:
            raise URLNotFoundError("OLLAMA_BASE_URL")
    
    elif args.mode == "offline" and args.model_engine == "vllm":
        MODEL_API_KEY = None
        MODEL_BASE_URL = os.getenv("VLLM_BASE_URL", None)

        if not MODEL_BASE_URL:
            raise URLNotFoundError("VLLM_BASE_URL")
        
    elif args.mode == "offline" and args.model_engine == "onnx":
        MODEL_BASE_URL = None
        MODEL_API_KEY = None

    elif args.mode == "offline" and args.model_engine == "huggingface":
        MODEL_API_KEY = None
        MODEL_BASE_URL = None
        # if not MODEL_BASE_URL:
        #     raise URLNotFoundError("VLLM_BASE_URL or OLLAMA_BASE_URL")
    else:
        raise ValueError(f"Unsupported model engine: {args.model_engine}")

    # Treat <=0 as "freestyle" (use a large default). Still keep a hard cap to avoid runaway.
    raw_max = int(getattr(args, "max_new_tokens", 256) or 0)
    if raw_max <= 0:
        effective_max = 1024
    else:
        effective_max = raw_max
    # Hard cap (safety): don't allow absurdly large generations from CLI.
    effective_max = min(max(effective_max, 64), 4096)

    llm = LLMs(
        type=args.mode,
        model_version=args.model_version,
        model_name=args.model_name,
        engine=args.model_engine,
        base_url=MODEL_BASE_URL,
        api_key=MODEL_API_KEY,
        max_tokens=effective_max,
    )

    # --- End Set up LLMs --- #

    # --- Relection Setup --- #

    # gpt = openai.OpenAI(api_key=os.getenv('OPEN_AI_KEY'))
    reflection = Reflection(llm=llm) if getattr(args, "enable_reflection", False) else None

    # --- End Reflection Setup --- #

    app = Flask(__name__)
    CORS(app)

    # Serve static HTML files
    @app.route('/')
    def serve_index():
        return send_from_directory('.', 'index.html')

    @app.route('/luyentap/')
    def serve_luyentap():
        return send_from_directory('luyentap', 'index.html')

    @app.route('/thidau/')
    def serve_thidau():
        return send_from_directory('thidau', 'index.html')

    # Legacy routes for backward compatibility
    @app.route('/game')
    def serve_game():
        return send_from_directory('luyentap', 'index.html')

    @app.route('/chatbot')
    def serve_chatbot():
        return send_from_directory('.', 'index.html')

    @app.route('/image/<path:filename>')
    def serve_image(filename):
        return send_from_directory('image', filename)

    # Initialize RAG
    if args.db == 'qdrant':
        QDRANT_API = os.getenv('QDRANT_API', None)
        QDRANT_URL = os.getenv('QDRANT_URL', None)
        if not QDRANT_API:
            raise APINotFoundError("QDRANT_API")
        if not QDRANT_URL:
            raise URLNotFoundError("QDRANT_URL")
        
        rag = RAG(
            type='qdrant',
            qdrant_api=QDRANT_API,
            qdrant_url=QDRANT_URL,
            embeddingName=args.embedding_model,
            embedding_backend=getattr(args, "embedding_backend", "sentence_transformers"),
            llm=llm,
        )

    elif args.db == 'mongodb':
        MONGODB_URI = os.getenv('MONGODB_URI')
        MONGODB_NAME = os.getenv('MONGODB_NAME')
        MONGODB_COLLECTION = os.getenv('MONGODB_COLLECTION')
        if not MONGODB_URI:
            raise URLNotFoundError("MONGODB_URI")
        if (not MONGODB_NAME) or (not MONGODB_COLLECTION):
            raise ValueNotFoundError(f"MONGODB_NAME and MONGODB_COLLECTION")
        
        rag = RAG(
            type='mongodb',
            mongodbUri=MONGODB_URI,
            dbName=MONGODB_NAME,
            dbCollection=MONGODB_COLLECTION,
            embeddingName=args.embedding_model,
            embedding_backend=getattr(args, "embedding_backend", "sentence_transformers"),
            llm=llm,
        )
    else:

        def chromadb_collection_exists(collection_name: str, persist_dir: str = "./chroma_db") -> bool:
            try:
                import chromadb
                client = chromadb.PersistentClient(path=persist_dir)
                collections = client.list_collections()
                return any(col.name == collection_name for col in collections)
            except Exception as e:
                print(f"Error checking ChromaDB collection: {e}")
                return False
        def csv_exists(folder_path: str) -> list:
            """
            Check if any CSV file exists in the given folder and return their file path.

            Args:
                folder_path (str): Path to the folder (e.g., "./data")

            Returns:
                list: list of csv files.    
            """
            if not os.path.isdir(folder_path):
                return []

            csv_paths = [
                os.path.abspath(os.path.join(folder_path, file))
                for file in os.listdir(folder_path)
                if file.lower().endswith(".csv") and os.path.isfile(os.path.join(folder_path, file))
            ]
            return csv_paths
        
        collection_name = f"{args.embedding_model.split('/')[-1]}__{getattr(args, 'embedding_backend', 'sentence_transformers')}"

        if not chromadb_collection_exists(collection_name=collection_name):
            csv_files = csv_exists(folder_path="data")
            if len(csv_files) == 0:
                raise DataNotFoundError
            else:
                print(f"The collection {collection_name} does not exist.\n")
                print("Starting to create new collection. Please make sure you have a valid CSV file in data folder.\n")
                print(f"Detected {len(csv_files)} csv files.\n")
                for i in  range(len(csv_files)):              
                    load_csv_to_chromadb(
                        csv_path=csv_files[i],
                        persist_dir="./chroma_db",
                        model_name=args.embedding_model,
                        embedding_backend=getattr(args, "embedding_backend", "sentence_transformers"),
                    )
                    print(f"Processed {i+1} files.\n")  
                print("The data insert process is complete.")

        rag = RAG(
            type='chromadb',
            embeddingName=args.embedding_model,
            embedding_backend=getattr(args, "embedding_backend", "sentence_transformers"),
            llm=llm
        )

    # --- Load MCQ index (for exact/near-exact match) ---
    # If the tagged MCQ file exists, use it (contains `topic`). Otherwise fall back to the raw MCQ file.
    mcq_index = []
    try:
        mcq_csv = None
        # Ưu tiên file clean mới tạo từ PDF
        if os.path.isfile("data/philosophy_mcq_clean.csv"):
            mcq_csv = "data/philosophy_mcq_clean.csv"
        elif os.path.isfile("data/philosophy_mcq_1000_tagged.csv"):
            mcq_csv = "data/philosophy_mcq_1000_tagged.csv"
        elif os.path.isfile("data/philosophy_mcq_1000.csv"):
            mcq_csv = "data/philosophy_mcq_1000.csv"

        if mcq_csv:
            print(f"[MCQ] Loading from {mcq_csv}")
            df_mcq = pd.read_csv(mcq_csv)
            # Pre-normalize for fast matching
            def _norm(s: str) -> str:
                s = (s or "").lower().strip()
                s = re.sub(r"\s+", " ", s)
                s = re.sub(r"[^0-9a-zàáạảãâầấậẩẫăằắặẩẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ ]", "", s)
                s = re.sub(r"\s+", " ", s)
                return s

            for _, r in df_mcq.iterrows():
                qtxt = str(r.get("question", ""))
                if not qtxt:
                    continue
                mcq_index.append(
                    {
                        "qid": int(r.get("qid")) if str(r.get("qid", "")).isdigit() else None,
                        "question": qtxt,
                        "question_norm": _norm(qtxt),
                        "combined_information": str(r.get("combined_information", "")),
                        "topic": str(r.get("topic", "")) if "topic" in df_mcq.columns else "",
                        # Lưu các đáp án A, B, C, D và đáp án đúng cho game
                        "A": str(r.get("A", "")).strip() if pd.notna(r.get("A")) else "",
                        "B": str(r.get("B", "")).strip() if pd.notna(r.get("B")) else "",
                        "C": str(r.get("C", "")).strip() if pd.notna(r.get("C")) else "",
                        "D": str(r.get("D", "")).strip() if pd.notna(r.get("D")) else "",
                        "answer": str(r.get("answer", "")).strip().upper() if pd.notna(r.get("answer")) else "",
                    }
                )
            print(f"[mcq] Loaded MCQ index from {mcq_csv}: {len(mcq_index)} questions")
    except Exception as e:
        print(f"[WARN] Failed to load MCQ index: {e}")

    # --- Maze game session store (in-memory) ---
    # NOTE: This is a simple single-process store. If you deploy multi-process, use Redis/db.
    game_sessions = {}
    GAME_ROOMS = 10

    def _game_hint_from_explanation(expl: str, correct: str) -> str:
        expl = (expl or "").strip()
        if expl:
            # Keep hints short for gameplay.
            expl = re.sub(r"\s+", " ", expl)
            return (expl[:220] + "…") if len(expl) > 220 else expl
        return f"Gợi ý: đáp án đúng là '{correct.upper()}'. Hãy đọc kỹ các lựa chọn và thử lại."

    def _pick_mcq_by_topic(topic: str | None, used_qids: set[int]) -> dict | None:
        """Pick a MCQ item from the loaded CSV index, optionally by topic, avoiding repeats."""
        if not mcq_index:
            return None
        candidates = mcq_index
        if topic:
            tnorm = (topic or "").strip().lower()
            candidates = [it for it in mcq_index if (it.get("topic") or "").strip().lower() == tnorm]
            if not candidates:
                # Fallback: contains match (user might type partial topic)
                candidates = [it for it in mcq_index if tnorm and tnorm in (it.get("topic") or "").strip().lower()]
        # Avoid repeats
        candidates = [it for it in candidates if it.get("qid") and int(it["qid"]) not in used_qids]
        # Chỉ cần ít nhất 2 đáp án (A và B) là được
        candidates = [it for it in candidates if it.get("A") and it.get("B")]
        if not candidates:
            return None
        # Random pick
        import random
        return random.choice(candidates)

    def _extract_mcq_fields_from_combined(ci: str) -> dict:
        """Best-effort extraction of question/options/answer from combined_information."""
        ci = (ci or "").replace("\r", "\n")
        # Question text
        m_q = re.search(r"C\s*[âa]u\s*(\d{1,4})\s*:\s*(.+?)(?=\n\s*A\.|\n\s*A\)|\n\s*A\:|\s+A\.)", ci, flags=re.IGNORECASE | re.DOTALL)
        qid_in_text = m_q.group(1) if m_q else ""
        qtext = (m_q.group(2).strip() if m_q else "").strip()
        # Options
        def opt(letter: str) -> str:
            mm = re.search(rf"\n\s*{letter}\s*[\.|\)|:]\s*(.+?)(?=\n\s*[ABCD]\s*[\.|\)|:]|\n\s*Đáp án đúng:|\n\s*Giải thích:|$)", ci, flags=re.IGNORECASE | re.DOTALL)
            return re.sub(r"\s+", " ", (mm.group(1).strip() if mm else "")).strip()

        A = opt("A")
        B = opt("B")
        C = opt("C")
        D = opt("D")
        # Answer
        m_ans = re.search(r"Đáp\s*án\s*đúng\s*:\s*([ABCD])", ci, flags=re.IGNORECASE)
        ans = (m_ans.group(1).upper() if m_ans else "")
        # Explanation
        m_ex = re.search(r"Giải\s*thích\s*:\s*(.+)$", ci, flags=re.IGNORECASE | re.DOTALL)
        expl = (m_ex.group(1).strip() if m_ex else "").strip()
        return {
            "qid_in_text": qid_in_text,
            "question": qtext,
            "A": A,
            "B": B,
            "C": C,
            "D": D,
            "answer": ans,
            "explanation": expl,
        }

    # --- MCQ helpers (for "1000 câu trắc nghiệm") ---
    def _extract_question_number(text: str):
        if not text:
            return None
        # Match: "câu 3", "Câu 003", "cho câu 12"...
        m = re.search(r"\bc\s*[âa]u\s*(\d{1,4})\b", text, flags=re.IGNORECASE)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def _extract_section_query(text: str) -> str:
        if not text:
            return ""
        # Simple heuristics: "phần X", "chương Y", "bài Z"
        m = re.search(r"\b(phần|chương|bài)\s*[:\-]?\s*([^\n\r]+)", text, flags=re.IGNORECASE)
        if not m:
            return ""
        # Keep short
        return (m.group(1) + " " + m.group(2)).strip()[:120]

    def _extract_topic_query(text: str) -> str:
        if not text:
            return ""
        # Accept: "chủ đề chân lý", "phần vật chất và ý thức", "cho câu hỏi về quy luật biện chứng"...
        t = text.strip()
        m = re.search(r"\b(chủ\s*đề)\s*[:\-]?\s*([^\n\r]+)", t, flags=re.IGNORECASE)
        if m:
            return m.group(2).strip()[:120]
        # If the user says "phần" but means semantic topic
        m2 = re.search(r"\b(câu hỏi về|phần|mục)\s+([^\n\r]+)", t, flags=re.IGNORECASE)
        if m2:
            return m2.group(2).strip()[:120]
        return ""

    def _looks_like_request_mcq(text: str) -> bool:
        """Heuristic classification for Multiple Choice (A/B/C/D).

        IMPORTANT: Do NOT treat any text containing "câu <n>" as MCQ by itself,
        because users will often reference question numbers in explanations.
        """
        t = (text or "").lower()

        # Strong MCQ signals: options A-D or explicit MCQ keywords.
        if re.search(r"\b(a\.|a\)|a\:)\s+.+\b(b\.|b\)|b\:)\s+.+\b(c\.|c\)|c\:)\s+.+\b(d\.|d\)|d\:)\s+", t, flags=re.IGNORECASE | re.DOTALL):
            return True
        if any(k in t for k in ["trắc nghiệm", "multiple choice", "chọn đáp án", "a.", "b.", "c.", "d."]):
            return True
        # Users sometimes paste a correction like "Câu trả lời đúng là: B".
        if re.search(r"câu\s+trả\s+lời\s+đúng\s+là\s*[:\-]?\s*[abcd]\b", t, flags=re.IGNORECASE):
            return True
        # "đáp án" alone can appear in user's pasted explanation; require question/option hints too.
        if "đáp án" in t and ("a" in t or "b" in t or "c" in t or "d" in t or "trắc nghiệm" in t):
            return True
        return False

    def _sanitize_assistant_output(text: str) -> str:
        """Remove template artifacts / garbage tokens from model outputs."""
        if not text:
            return ""
        t = str(text)
        # Remove common chat-template artifacts
        t = re.sub(r"<\|im_start\|>\s*\w+\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"<\|im_end\|>", "", t, flags=re.IGNORECASE)
        # Collapse excessive whitespace
        t = t.replace("\r", "\n")
        t = re.sub(r"\n{3,}", "\n\n", t)
        t = re.sub(r"[ \t]{2,}", " ", t)
        return t.strip()

    def _clean_user_query_for_generation(text: str) -> str:
        """Remove meta references that the user doesn't want repeated.

        Example to remove: "Đây là khái niệm được đề cập ở câu 205..., đáp án D".
        We ONLY apply this cleaning for non-MCQ generation.
        """
        if not text:
            return ""
        t = str(text).strip()
        # Drop common meta lines/clauses about question numbers and answers.
        patterns = [
            r"\b(đây\s+là\s+khái\s+niệm\s+được\s+đề\s+cập\s+ở\s+câu\s*\d{1,4}[^\n\.]*\.?)(\s|$)",
            r"\b(đề\s+cập\s+ở\s+câu\s*\d{1,4}[^\n\.]*\.?)(\s|$)",
            r"\b(câu\s*\d{1,4}\s+của\s+giáo\s+trình[^\n\.]*\.?)(\s|$)",
            r"\b(đáp\s+án\s+([abcd]|đúng\s+là\s*[abcd])\b[^\n\.]*\.?)(\s|$)",
        ]
        for p in patterns:
            t = re.sub(p, " ", t, flags=re.IGNORECASE)
        # If user pasted a long paragraph, keep only the last question-ish line if available.
        # (This avoids feeding the model a long essay and reduces parroting.)
        lines = [ln.strip() for ln in re.split(r"\r?\n", t) if ln.strip()]
        if len(lines) >= 2:
            # Prefer the last line that ends with '?' or contains "là" patterns.
            for ln in reversed(lines):
                if "?" in ln:
                    t = ln
                    break
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _format_mcq_from_doc(doc: str) -> str:
        """Extract a clean MCQ block from combined_information."""
        if not doc:
            return ""
        # Best-effort parse based on the string we stored in combined_information.
        # Example: "... Câu 3: ... A. ... B. ... C. ... D. ... Đáp án đúng: A. ..."
        # We'll return a neat block.
        doc = doc.replace("\r", "\n")
        # Try to find the "Câu" block
        m = re.search(r"(C\s*[âa]u\s*\d{1,4}\s*:\s*.+)", doc, flags=re.IGNORECASE)
        core = m.group(1).strip() if m else doc.strip()

        # Normalize spacing
        core = re.sub(r"\s+", " ", core)
        # Re-insert line breaks before options and answer
        core = re.sub(r"\s+(A\.)\s+", r"\n\1 ", core)
        core = re.sub(r"\s+(B\.)\s+", r"\n\1 ", core)
        core = re.sub(r"\s+(C\.)\s+", r"\n\1 ", core)
        core = re.sub(r"\s+(D\.)\s+", r"\n\1 ", core)
        core = re.sub(r"\s+(Đáp án đúng:)\s+", r"\n\1 ", core, flags=re.IGNORECASE)
        # Loại bỏ phần "Giải thích: Câu X của giáo trình..." - không nên đề cập số câu
        core = re.sub(r"\s*Giải thích:.*$", "", core, flags=re.IGNORECASE)
        return core.strip()

    def _normalize_topic_name(topic_q: str) -> str:
        """Map user topic phrases to the canonical topic names in the tagged dataset."""
        t = (topic_q or "").lower()
        t = re.sub(r"\s+", " ", t).strip()
        mapping = [
            ("vật chất", "Vật chất & Ý thức"),
            ("ý thức", "Vật chất & Ý thức"),
            ("nhận thức", "Nguồn gốc & bản chất nhận thức"),
            ("nguồn gốc nhận thức", "Nguồn gốc & bản chất nhận thức"),
            ("chân lý", "Chân lý"),
            ("quy luật", "Các quy luật biện chứng"),
            ("biện chứng", "Các quy luật biện chứng"),
            ("phạm trù", "Phạm trù phép biện chứng"),
            ("giai cấp", "Xã hội, giai cấp, nhà nước"),
            ("nhà nước", "Xã hội, giai cấp, nhà nước"),
            ("xã hội", "Xã hội, giai cấp, nhà nước"),
        ]
        for key, canonical in mapping:
            if key in t:
                return canonical
        # If user already typed canonical-ish
        return topic_q.strip()

    def _try_exact_mcq_match(user_text: str) -> str:
        """If the user provided an (almost) exact MCQ question, return the MCQ block immediately.
        
        Đây là cache để trả lời nhanh khi user hỏi câu giống hệt trong database.
        Không cần gọi LLM, trả về ngay từ CSV.
        """
        if not user_text or not mcq_index:
            return ""
        # Normalize user text and strip typical prefixes.
        ut = (user_text or "").strip()
        ut = re.sub(r"^\s*(c\s*[âa]u\s*\d{1,4}\s*[:\-\.]\s*)", "", ut, flags=re.IGNORECASE)

        def _norm_user(s: str) -> str:
            s = s.lower().strip()
            s = re.sub(r"\s+", " ", s)
            s = re.sub(r"[^0-9a-zàáạảãâầấậẩẫăằắặẩẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ ]", "", s)
            s = re.sub(r"\s+", " ", s)
            return s

        u = _norm_user(ut)
        if len(u) < 12:
            return ""

        # Heuristic: exact equality or containment with enough length.
        for item in mcq_index:
            qn = item["question_norm"]
            if not qn:
                continue
            if u == qn or (u in qn or qn in u) and min(len(u), len(qn)) >= 25:
                # Tìm thấy câu hỏi giống hệt - trả về từ cache CSV luôn
                q = item.get("question", "")
                a = item.get("A", "")
                b = item.get("B", "")
                c = item.get("C", "")
                d = item.get("D", "")
                ans = item.get("answer", "").upper()
                
                # Format câu trả lời
                out = f"📝 {q}"
                if a:
                    out += f"\nA. {a}"
                if b:
                    out += f"\nB. {b}"
                if c:
                    out += f"\nC. {c}"
                if d:
                    out += f"\nD. {d}"
                if ans:
                    out += f"\n\n✅ Đáp án đúng: {ans}"
                return out
        return ""
    # Initialize ReRanker (optional). CrossEncoder models can be slow to download/start.
    reranker = None
    if getattr(args, "enable_rerank", True):
        reranker = Reranker(model_name=args.reranker)

    def process_query(query):
        return query.lower()

    @app.route('/', methods=['GET'])
    def home():
        return jsonify({
            "status": "ok",
            "message": "RAG server is running. Use POST /api/search with JSON chat messages.",
            "example": [{"role": "user", "content": "Triết học là gì?"}],
        })

    @app.route('/favicon.ico', methods=['GET'])
    def favicon():
        # Avoid 404 noise in browser logs.
        return ("", 204)

    @app.route('/api/search', methods=['GET', 'POST'])
    def handle_query():

        # Helpful message when someone opens /api/search in a browser (GET).
        if request.method == 'GET':
            return jsonify({
                "status": "ok",
                "message": "Use POST /api/search with JSON messages. GET is for this help only.",
                "example": [{"role": "user", "content": "Triết học là gì?"}],
            })

        print("\n🚀 Starting RAG Server with the following setup:")
        print("===============================================")
        print(f"🔧 Mode: {args.mode}")
        print(f"🤖 Model Name: {args.model_name}")
        print(f"🛠️ Model Engine: {args.model_engine}")
        print(f"📦 Model Version: {args.model_version}")
        print(f"🧠 Embedding Model: {args.embedding_model}")
        print(f"📊 Reranker Model: {args.reranker}")
        print(f"🗃️ Vector DB: {args.db}")

        
        data = request.get_json(silent=True)
        if not data:
            return jsonify({
                "error": "Missing JSON body. Send a JSON array of chat messages.",
                "example": [{"role": "user", "content": "Triết học là gì?"}],
            }), 400
        data = list(data)

        # Get latest user message text
        user_messages = [m for m in data if isinstance(m, dict) and m.get("role") == "user" and m.get("content")]
        last_user_text = user_messages[-1]["content"] if user_messages else ""

        # Initialize query_text early so all fallbacks are safe.
        query_text_raw = (last_user_text or "").strip()
        is_mcq_user = _looks_like_request_mcq(query_text_raw)
        query_text = query_text_raw

        # Fast path: if the user pasted an MCQ question exactly (or near-exact),
        # return the MCQ (with correct answer) immediately without routing/LLM.
        if is_mcq_user:
            exact_mcq = _try_exact_mcq_match(query_text)
            if exact_mcq:
                return jsonify({'content': exact_mcq, 'role': 'assistant'})

        # Reflection should output a standalone question text.
        # IMPORTANT: Reflection may return a non-string (e.g., dict) depending on LLM wrapper.
        # Always normalize to a clean string for routing.
        # Reflection (query rewriting) adds extra LLM latency. Keep it optional.
        if reflection is not None:
            try:
                reflected_query = reflection(data)
            except Exception as e:
                print(f"[WARN] Reflection failed, falling back to user query. Error: {e}")
                reflected_query = last_user_text
        else:
            reflected_query = last_user_text
        if isinstance(reflected_query, str):
            query_text = reflected_query.strip()
        else:
            query_text = (last_user_text or "").strip()

        # For non-MCQ questions, clean out "câu N / đáp án" meta so we don't echo it.
        if not is_mcq_user:
            query_text = _clean_user_query_for_generation(query_text)

        # If reflection failed (Gemini 404 / auth / quota) it may return an error message.
        # Never route based on that error text; route based on the user's real question.
        if query_text.startswith("❌") or query_text.lower().startswith("error"):
            query_text = (last_user_text or query_text).strip()

        # Safety: always send a string into the semantic router.
        guidedRoute = semanticRouter.guide(query_text)[1]
        print(f"[debug] query_text: {query_text!r}")
        print(f"[debug] router guidedRoute (before keyword override): {guidedRoute!r}")

        # Extra robustness: obvious philosophy keywords should never be rejected.
        qt_lower = query_text.lower()

        # Extra robustness: obvious philosophy keywords should never be rejected.
        # If the question contains "triết học" at all, we force the philosophy route.
        philosophy_keywords = [
            "triết học",
            "mác",
            "lênin",
            "duy vật",
            "biện chứng",
            "vật chất",
            "ý thức",
            "thực tiễn",
            "chân lý",
        ]
        if "triết học" in qt_lower or any(k in qt_lower for k in philosophy_keywords):
            guidedRoute = PRODUCT_ROUTE_NAME

        # Retrieval-based fallback: some short philosophy questions (e.g. "Thực tiễn là gì?")
        # can be misclassified by the semantic router. If the vector store returns
        # relevant hits, we treat it as philosophy to avoid wrongful refusal.
        if guidedRoute != PRODUCT_ROUTE_NAME:
            try:
                probe_hits = rag.vector_search(query_text, limit=1)
                if isinstance(probe_hits, list) and len(probe_hits) > 0:
                    # Score is similarity (1-distance). Use a modest threshold.
                    score = probe_hits[0].get("score", 0.0)
                    if score is None:
                        score = 0.0
                    if float(score) >= 0.25:
                        guidedRoute = PRODUCT_ROUTE_NAME
            except Exception as e:
                print(f"[WARN] retrieval fallback probe failed: {e}")

        print(f"[debug] guidedRoute (after keyword override): {guidedRoute!r}")

        if guidedRoute == PRODUCT_ROUTE_NAME:
            # Guide to RAG system
            print("Guide to RAGs - Philosophy Knowledge Base")

            # --- Dedicated MCQ behavior ---
            # If user asks for a specific question number or requests questions by section,
            # we return the exact MCQ content (including correct answer) rather than a free-form explanation.
            qnum = _extract_question_number(query_text)
            section_q = _extract_section_query(query_text)
            topic_q = _extract_topic_query(query_text)

            if qnum is not None or topic_q or (section_q and ("đưa" in query_text.lower() or "cho" in query_text.lower() or "câu hỏi" in query_text.lower())) or _looks_like_request_mcq(query_text):
                # Build a retrieval query focused on MCQ items.
                if qnum is not None:
                    mcq_query = f"Câu {qnum}:"
                elif topic_q:
                    canonical_topic = _normalize_topic_name(topic_q)
                    mcq_query = f"Chủ đề: {canonical_topic}. Câu hỏi trắc nghiệm"
                elif section_q:
                    mcq_query = f"Phần: {section_q}. Hãy đưa các câu hỏi trắc nghiệm thuộc phần này"
                else:
                    mcq_query = query_text

                top_k_mcq = getattr(args, "top_k_mcq", 5)
                hits = rag.vector_search(mcq_query, limit=top_k_mcq)

                # Filter to likely MCQ docs
                mcq_docs = []
                for h in hits:
                    doc = h.get('combined_information', '')
                    if not doc:
                        continue
                    if "Loại: Trắc nghiệm" in doc or "Đáp án đúng:" in doc or re.search(r"\bA\.|\bB\.|\bC\.|\bD\.", doc):
                        mcq_docs.append(doc)

                # If asking by question number, try to pick the exact one.
                if qnum is not None:
                    exact = None
                    for d in mcq_docs:
                        if re.search(rf"C\s*[âa]u\s*{qnum}\s*:\s*", d, flags=re.IGNORECASE):
                            exact = d
                            break
                    if exact is None and len(hits) > 0:
                        exact = hits[0].get('combined_information', '')

                    if exact:
                        response = _format_mcq_from_doc(exact)
                    else:
                        response = f"Mình chưa tìm thấy dữ liệu cho Câu {qnum}. Bạn thử hỏi lại theo dạng 'Câu {qnum}:' hoặc gửi thêm vài từ trong câu hỏi nhé."

                # If asking by topic/section, return a list (top N)
                elif topic_q or section_q:
                    label = topic_q or section_q
                    if len(mcq_docs) == 0:
                        response = f"Mình chưa tìm thấy câu hỏi trắc nghiệm cho '{label}'. Bạn thử ghi rõ hơn (ví dụ: 'Vật chất & Ý thức', 'Quy luật biện chứng', 'Nhận thức', 'Chân lý') hoặc gửi 1 câu số thuộc phần đó nhé."
                    else:
                        out = [f"Các câu hỏi trắc nghiệm cho '{label}' (kèm đáp án đúng):"]
                        for i, d in enumerate(mcq_docs[:top_k_mcq], start=1):
                            out.append(f"\n---\n{i}) {_format_mcq_from_doc(d)}")
                        response = "\n".join(out)

                # Generic MCQ request: return best match (with answer)
                else:
                    if len(mcq_docs) == 0 and len(hits) > 0:
                        response = _format_mcq_from_doc(hits[0].get('combined_information', ''))
                    elif len(mcq_docs) > 0:
                        response = _format_mcq_from_doc(mcq_docs[0])
                    else:
                        response = "Mình chưa tìm thấy câu trắc nghiệm phù hợp. Bạn thử gửi nguyên văn câu hỏi hoặc kèm 'Câu số ...' nhé."

                return jsonify({'content': response, 'role': 'assistant'})

            # Take relevant documents from RAG system
            top_k = getattr(args, "top_k", 2)
            passages = [passage['combined_information'] for passage in rag.vector_search(query_text, limit=top_k)]

            # Truncate passages to keep the prompt short => faster generation.
            max_chars = getattr(args, "passage_max_chars", 900)
            if max_chars and max_chars > 0:
                passages = [p[:max_chars] for p in passages]
            
            # Re-rank retrieved documents (optional)
            if reranker is not None:
                scores, ranked_passages = reranker(query_text, passages)
            else:
                ranked_passages = passages
            source_information = ""
            for i in range(len(ranked_passages)):
                source_information += f"{i+1}. {ranked_passages[i]}\n\n"

            combined_information = f"""Bạn là một chuyên gia triết học, giảng viên giảng dạy môn Triết học Mác-Lênin. 
Nhiệm vụ của bạn là trả lời các câu hỏi về triết học dựa trên giáo trình triết học.

QUY TẮC NGHIÊM NGẶT:
- CHỈ trả lời các câu hỏi liên quan đến triết học, triết học Mác-Lênin
- KHÔNG trả lời về: lập trình, toán học, vật lý, hóa học, y học, pháp luật, kinh tế (trừ kinh tế chính trị), công nghệ, điện thoại, sản phẩm, thể thao, giải trí, hoặc BẤT KỲ chủ đề nào NGOÀI triết học
- Nếu câu hỏi KHÔNG liên quan đến triết học, BẮT BUỘC phải từ chối và chỉ trả lời: "Xin lỗi, tôi chỉ có thể trả lời các câu hỏi về Triết học Mác-Lênin dựa trên giáo trình. Câu hỏi của bạn nằm ngoài phạm vi chuyên môn của tôi. Bạn có câu hỏi nào về triết học không?"
- CHỈ sử dụng thông tin từ tài liệu tham khảo dưới đây, KHÔNG tự bịa đặt
- Trả lời bằng tiếng Việt, rõ ràng, có căn cứ từ giáo trình

Câu hỏi của sinh viên: {query_text}

Tài liệu tham khảo từ giáo trình triết học:
{source_information}

Hãy phân tích xem câu hỏi có liên quan đến triết học không. Nếu có, trả lời dựa trên tài liệu. Nếu không, từ chối lịch sự:"""
            data.append({
                "role": "user",
                "content": combined_information
            })
            response = rag.generate_content(data)
        else:
            # Câu hỏi không liên quan đến triết học -> Từ chối
            print("Guide to LLMs - Non-philosophy question detected")
            response = "Xin lỗi, tôi chỉ có thể trả lời các câu hỏi về Triết học Mác-Lênin dựa trên giáo trình. Câu hỏi của bạn nằm ngoài phạm vi chuyên môn của tôi. Bạn có câu hỏi nào về triết học không?"
        
        return jsonify({
            'content': _sanitize_assistant_output(response),
            'role': 'assistant'
            })

    @app.route('/api/search/stream', methods=['POST'])
    def handle_query_stream():
        """Server-Sent Events streaming endpoint.

        Goal: "cảm giác nhanh ngay lập tức" while keeping correctness.
        We stream progress events quickly, then stream the final answer.
        (True token-by-token streaming requires an LLM backend that supports
        streaming callbacks; Ollama can, but we keep this robust/simple.)
        """

        def sse(event: str, data: str) -> str:
            safe = (data or "").replace("\r", "").replace("\n", "\\n")
            return f"event: {event}\ndata: {safe}\n\n"

        data = request.get_json(silent=True)
        if not data:
            return Response(
                sse("error", "Missing JSON body"),
                mimetype="text/event-stream",
            )
        data = list(data)

        user_messages = [m for m in data if isinstance(m, dict) and m.get("role") == "user" and m.get("content")]
        last_user_text = user_messages[-1]["content"] if user_messages else ""
        query_text = (last_user_text or "").strip()

        def generate():
            yield sse("status", "retrieval")

            # Optional reflection (disabled by default for speed)
            if reflection is not None:
                try:
                    rq = reflection(data)
                    if isinstance(rq, str) and rq.strip():
                        qt = rq.strip()
                    else:
                        qt = query_text
                except Exception:
                    qt = query_text
            else:
                qt = query_text

            if qt.startswith("❌") or qt.lower().startswith("error"):
                qt = query_text

            yield sse("status", "routing")
            guidedRoute = semanticRouter.guide(qt)[1]

            # Keyword override for philosophy
            qt_lower = qt.lower()
            philosophy_keywords = [
                "triết học",
                "mác",
                "lênin",
                "biện chứng",
                "duy vật",
                "duy tâm",
                "ý thức",
                "vật chất",
            ]
            if any(k in qt_lower for k in philosophy_keywords):
                guided = 'products'
            else:
                guided = guidedRoute

            # Retrieval-based fallback for short philosophy questions misrouted by router.
            if guided != 'products':
                try:
                    probe_hits = rag.vector_search(qt, limit=1)
                    if isinstance(probe_hits, list) and len(probe_hits) > 0:
                        score = probe_hits[0].get("score", 0.0)
                        if score is None:
                            score = 0.0
                        if float(score) >= 0.25:
                            guided = 'products'
                except Exception as e:
                    yield sse("status", f"warn:retrieval_probe_failed")
                    print(f"[WARN] retrieval fallback probe failed: {e}")

            if guided != 'products':
                refusal = "Xin lỗi, tôi chỉ có thể trả lời các câu hỏi về Triết học Mác-Lênin dựa trên giáo trình. Câu hỏi của bạn nằm ngoài phạm vi chuyên môn của tôi. Bạn có câu hỏi nào về triết học không?"
                yield sse("final", refusal)
                return

            yield sse("status", "generating")

            # Retrieval
            top_k = getattr(args, "top_k", 2)
            passages = [p['combined_information'] for p in rag.vector_search(qt, limit=top_k)]
            max_chars = getattr(args, "passage_max_chars", 900)
            if max_chars and max_chars > 0:
                passages = [p[:max_chars] for p in passages]

            source_information = ""
            for i, p in enumerate(passages, start=1):
                source_information += f"{i}. {p}\n\n"

            combined_information = f"""Bạn là một chuyên gia triết học, giảng viên giảng dạy môn Triết học Mác-Lênin.
Nhiệm vụ của bạn là trả lời các câu hỏi về triết học dựa trên giáo trình triết học.

QUY TẮC NGHIÊM NGẶT:
- CHỈ trả lời các câu hỏi liên quan đến triết học, triết học Mác-Lênin
- Nếu câu hỏi KHÔNG liên quan đến triết học, BẮT BUỘC phải từ chối theo mẫu.
- CHỈ sử dụng thông tin từ tài liệu tham khảo dưới đây, KHÔNG tự bịa đặt
- Trả lời bằng tiếng Việt, rõ ràng, có căn cứ từ giáo trình

Câu hỏi của sinh viên: {qt}

Tài liệu tham khảo từ giáo trình triết học:
{source_information}

Hãy trả lời ngắn gọn, đúng trọng tâm, có trích ý chính từ tài liệu (không bịa)."""

            prompt = list(data)
            prompt.append({"role": "user", "content": combined_information})

            try:
                # True token streaming when supported (Ollama).
                llm = getattr(rag, "llm", None)
                stream_fn = getattr(llm, "stream_content", None) if llm is not None else None
                if callable(stream_fn):
                    chunks = []
                    for chunk in stream_fn(prompt):
                        if not chunk:
                            continue
                        chunks.append(chunk)
                        yield sse("chunk", chunk)
                    answer = "".join(chunks)
                else:
                    answer = rag.generate_content(prompt)
            except Exception as e:
                yield sse("error", f"Generation failed: {e}")
                return

            yield sse("final", answer)

        return Response(generate(), mimetype="text/event-stream")

    # -------------------- Competition Rooms API --------------------
    import random
    import string
    
    # Store competition rooms: room_code -> room_data
    competition_rooms = {}
    
    def generate_room_code():
        """Generate a 6-character room code (uppercase letters + digits)."""
        chars = string.ascii_uppercase + string.digits
        while True:
            code = ''.join(random.choices(chars, k=6))
            if code not in competition_rooms:
                return code
    
    @app.route('/api/competition/create', methods=['POST'])
    def create_competition_room():
        """Create a new competition room. Returns room code."""
        body = request.get_json(silent=True) or {}
        host_name = (body.get('host_name') or 'Host').strip()[:20]
        room_name = (body.get('room_name') or 'Phòng Thi Đấu').strip()[:30]
        
        room_code = generate_room_code()
        competition_rooms[room_code] = {
            "code": room_code,
            "room_name": room_name,
            "host": host_name,
            "players": {host_name: {"name": host_name, "is_host": True, "ready": False, "completed": False, "correct": 0, "wrong": 0, "time_ms": 0, "rank": 0}},
            "status": "waiting",  # waiting, playing, finished
            "created_at": time.time(),
            "started_at": None,
            "game_sessions": {},  # player_name -> game session
        }
        return jsonify({
            "status": "ok",
            "room_code": room_code,
            "room_name": room_name,
            "host": host_name,
            "message": f"Phòng {room_code} đã được tạo!"
        })
    
    @app.route('/api/competition/join', methods=['POST'])
    def join_competition_room():
        """Join an existing competition room."""
        body = request.get_json(silent=True) or {}
        room_code = (body.get('room_code') or '').strip().upper()
        player_name = (body.get('player_name') or '').strip()[:20]
        
        if not room_code or room_code not in competition_rooms:
            return jsonify({"status": "error", "message": "Mã phòng không tồn tại!"}), 404
        
        if not player_name:
            return jsonify({"status": "error", "message": "Vui lòng nhập tên!"}), 400
        
        room = competition_rooms[room_code]
        
        if room["status"] != "waiting":
            return jsonify({"status": "error", "message": "Phòng đã bắt đầu hoặc kết thúc!"}), 400
        
        if player_name in room["players"]:
            return jsonify({"status": "error", "message": "Tên đã được sử dụng!"}), 400
        
        if len(room["players"]) >= 50:
            return jsonify({"status": "error", "message": "Phòng đã đầy!"}), 400
        
        room["players"][player_name] = {
            "name": player_name,
            "is_host": False,
            "ready": False,
            "completed": False,
            "correct": 0,
            "wrong": 0,
            "time_ms": 0,
            "rank": 0
        }
        
        return jsonify({
            "status": "ok",
            "room_code": room_code,
            "player_name": player_name,
            "players": list(room["players"].values()),
            "message": f"Đã tham gia phòng {room_code}!"
        })
    
    @app.route('/api/competition/room/<room_code>', methods=['GET'])
    def get_competition_room(room_code):
        """Get room status and player list."""
        room_code = room_code.upper()
        if room_code not in competition_rooms:
            return jsonify({"status": "error", "message": "Phòng không tồn tại!"}), 404
        
        room = competition_rooms[room_code]
        return jsonify({
            "status": "ok",
            "room_code": room_code,
            "room_name": room.get("room_name", "Phòng Thi Đấu"),
            "host": room["host"],
            "room_status": room["status"],
            "players": list(room["players"].values()),
            "player_count": len(room["players"]),
            "started_at": room["started_at"],
        })
    
    @app.route('/api/competition/start', methods=['POST'])
    def start_competition():
        """Host starts the competition."""
        body = request.get_json(silent=True) or {}
        room_code = (body.get('room_code') or '').strip().upper()
        host_name = (body.get('host_name') or '').strip()
        
        if room_code not in competition_rooms:
            return jsonify({"status": "error", "message": "Phòng không tồn tại!"}), 404
        
        room = competition_rooms[room_code]
        
        if room["host"] != host_name:
            return jsonify({"status": "error", "message": "Chỉ host mới được bắt đầu!"}), 403
        
        if room["status"] != "waiting":
            return jsonify({"status": "error", "message": "Phòng đã bắt đầu!"}), 400
        
        if len(room["players"]) < 1:
            return jsonify({"status": "error", "message": "Cần ít nhất 1 người chơi!"}), 400
        
        room["status"] = "playing"
        room["started_at"] = time.time()
        
        # Create game session for each player
        for player_name in room["players"]:
            sid = str(uuid.uuid4())
            room["game_sessions"][player_name] = sid
            game_sessions[sid] = {
                "room": 1,
                "used_qids": set(),
                "current": None,
                "created_at": time.time(),
                "competition_room": room_code,
                "player_name": player_name,
            }
        
        return jsonify({
            "status": "ok",
            "room_code": room_code,
            "room_status": "playing",
            "message": "Cuộc thi đã bắt đầu!",
            "sessions": {name: room["game_sessions"][name] for name in room["players"]}
        })
    
    @app.route('/api/competition/finish', methods=['POST'])
    def finish_competition_player():
        """Mark a player as finished and update ranking."""
        body = request.get_json(silent=True) or {}
        room_code = (body.get('room_code') or '').strip().upper()
        player_name = (body.get('player_name') or '').strip()
        correct = int(body.get('correct', 0))
        wrong = int(body.get('wrong', 0))
        time_ms = int(body.get('time_ms', 0))
        
        if room_code not in competition_rooms:
            return jsonify({"status": "error", "message": "Phòng không tồn tại!"}), 404
        
        room = competition_rooms[room_code]
        
        if player_name not in room["players"]:
            return jsonify({"status": "error", "message": "Người chơi không trong phòng!"}), 404
        
        # Update player stats
        player = room["players"][player_name]
        player["completed"] = True
        player["correct"] = correct
        player["wrong"] = wrong
        player["time_ms"] = time_ms
        
        # Calculate rankings (sort by correct DESC, then time ASC)
        completed_players = [p for p in room["players"].values() if p["completed"]]
        completed_players.sort(key=lambda x: (-x["correct"], x["time_ms"]))
        
        for i, p in enumerate(completed_players):
            room["players"][p["name"]]["rank"] = i + 1
        
        # Check if all players finished
        all_finished = all(p["completed"] for p in room["players"].values())
        if all_finished:
            room["status"] = "finished"
        
        return jsonify({
            "status": "ok",
            "room_status": room["status"],
            "player_rank": player["rank"],
            "rankings": sorted(
                [{"name": p["name"], "correct": p["correct"], "wrong": p["wrong"], "time_ms": p["time_ms"], "rank": p["rank"]} 
                 for p in room["players"].values() if p["completed"]],
                key=lambda x: x["rank"]
            )
        })
    
    @app.route('/api/competition/rankings/<room_code>', methods=['GET'])
    def get_competition_rankings(room_code):
        """Get current rankings for a room."""
        room_code = room_code.upper()
        if room_code not in competition_rooms:
            return jsonify({"status": "error", "message": "Phòng không tồn tại!"}), 404
        
        room = competition_rooms[room_code]
        rankings = sorted(
            [{"name": p["name"], "correct": p["correct"], "wrong": p["wrong"], "time_ms": p["time_ms"], "rank": p["rank"], "completed": p["completed"]} 
             for p in room["players"].values()],
            key=lambda x: (not x["completed"], x["rank"] if x["rank"] > 0 else 9999)
        )
        
        return jsonify({
            "status": "ok",
            "room_code": room_code,
            "room_status": room["status"],
            "rankings": rankings,
            "total_players": len(room["players"]),
            "completed_count": sum(1 for p in room["players"].values() if p["completed"])
        })

    @app.route('/api/competition/progress', methods=['POST'])
    def update_player_progress():
        """Update player's current progress (correct/wrong count) in real-time."""
        body = request.get_json(silent=True) or {}
        room_code = (body.get('room_code') or '').strip().upper()
        player_name = (body.get('player_name') or '').strip()
        correct = int(body.get('correct', 0))
        wrong = int(body.get('wrong', 0))
        
        if room_code not in competition_rooms:
            return jsonify({"status": "error", "message": "Phòng không tồn tại!"}), 404
        
        room = competition_rooms[room_code]
        
        if player_name not in room["players"]:
            return jsonify({"status": "error", "message": "Người chơi không trong phòng!"}), 404
        
        # Update player's progress (without marking as completed)
        player = room["players"][player_name]
        player["correct"] = correct
        player["wrong"] = wrong
        
        return jsonify({
            "status": "ok",
            "correct": correct,
            "wrong": wrong
        })

    @app.route('/api/competition/live/<room_code>', methods=['GET'])
    def get_live_scoreboard(room_code):
        """Get live scoreboard for host to see all players' progress."""
        room_code = room_code.upper()
        if room_code not in competition_rooms:
            return jsonify({"status": "error", "message": "Phòng không tồn tại!"}), 404
        
        room = competition_rooms[room_code]
        
        # Get all players (excluding host) with their current progress
        players = []
        for p in room["players"].values():
            if not p.get("is_host", False):
                players.append({
                    "name": p["name"],
                    "correct": p.get("correct", 0),
                    "wrong": p.get("wrong", 0),
                    "time_ms": p.get("time_ms", 0) if p.get("completed", False) else None,
                    "completed": p.get("completed", False)
                })
        
        return jsonify({
            "status": "ok",
            "room_code": room_code,
            "room_status": room["status"],
            "players": players,
            "total_players": len(players),
            "completed_count": sum(1 for p in players if p["completed"])
        })

    @app.route('/api/competition/end', methods=['POST'])
    def end_competition():
        """Host ends the competition (force finish all players)."""
        body = request.get_json(silent=True) or {}
        room_code = (body.get('room_code') or '').strip().upper()
        host_name = (body.get('host_name') or '').strip()
        
        if room_code not in competition_rooms:
            return jsonify({"status": "error", "message": "Phòng không tồn tại!"}), 404
        
        room = competition_rooms[room_code]
        
        if room["host"] != host_name:
            return jsonify({"status": "error", "message": "Chỉ host mới được kết thúc!"}), 403
        
        # Mark room as finished
        room["status"] = "finished"
        
        # Calculate rankings for all players (even incomplete ones)
        all_players = list(room["players"].values())
        all_players.sort(key=lambda x: (-x.get("correct", 0), x.get("time_ms", float('inf')) if x.get("completed") else float('inf')))
        
        for i, p in enumerate(all_players):
            room["players"][p["name"]]["rank"] = i + 1
        
        return jsonify({
            "status": "ok",
            "room_status": "finished",
            "message": "Trận đấu đã kết thúc!"
        })

    # -------------------- Maze Game API --------------------
    @app.route('/api/game/maze/start', methods=['POST'])
    def maze_start():
        """Create a new 10-room maze session."""
        sid = str(uuid.uuid4())
        game_sessions[sid] = {
            "room": 1,
            "used_qids": set(),
            "current": None,
            "created_at": time.time(),
        }
        return jsonify({
            "session_id": sid,
            "room": 1,
            "rooms_total": GAME_ROOMS,
            "message": "Bạn đã bước vào Mê Cung Triết Học. Hãy chọn chủ đề cho Phòng 1!",
        })

    @app.route('/api/game/maze/question', methods=['POST'])
    def maze_question():
        """Get the current room question (or generate a new one) by topic."""
        body = request.get_json(silent=True) or {}
        sid = body.get('session_id')
        topic = (body.get('topic') or '').strip()
        if not sid or sid not in game_sessions:
            return jsonify({"error": "Invalid session_id. Call /api/game/maze/start first."}), 400

        s = game_sessions[sid]
        if s["room"] > GAME_ROOMS:
            return jsonify({"done": True, "message": "Bạn đã thoát khỏi mê cung rồi!"})

        used = s["used_qids"]
        item = _pick_mcq_by_topic(topic if topic else None, used)
        if not item:
            return jsonify({"error": f"Không tìm thấy câu hỏi cho chủ đề '{topic}'."}), 404

        # Lấy trực tiếp từ CSV (file philosophy_mcq_clean.csv có format chuẩn)
        qid = int(item.get("qid") or 0)
        question_text = str(item.get("question") or "").strip()
        opt_A = str(item.get("A") or "").strip()
        opt_B = str(item.get("B") or "").strip()
        opt_C = str(item.get("C") or "").strip()
        opt_D = str(item.get("D") or "").strip()
        answer_letter = str(item.get("answer") or "").strip().upper()
        
        s["current"] = {
            "qid": qid,
            "answer": answer_letter,
            "topic": item.get("topic", ""),
            "question": question_text,
        }
        used.add(qid)

        return jsonify({
            "status": "ok",
            "session_id": sid,
            "room": s["room"],
            "rooms_total": GAME_ROOMS,
            "topic": item.get("topic", ""),
            "qid": qid,
            "question": question_text,
            "A": opt_A,
            "B": opt_B,
            "C": opt_C,
            "D": opt_D,
        })

    @app.route('/api/game/maze/answer', methods=['POST'])
    def maze_answer():
        """Submit an answer for the current room; returns correct/incorrect + hint."""
        body = request.get_json(silent=True) or {}
        sid = body.get('session_id')
        # Accept both 'choice' and 'answer' keys for flexibility
        choice = (body.get('choice') or body.get('answer') or '').strip().upper()
        if not sid or sid not in game_sessions:
            return jsonify({"error": "Invalid session_id."}), 400
        if choice not in {"A", "B", "C", "D"}:
            return jsonify({"error": "answer must be one of A/B/C/D"}), 400

        s = game_sessions[sid]
        cur = s.get("current")
        if not cur:
            return jsonify({"error": "No active question. Call /api/game/maze/question first."}), 400

        correct = (cur.get("answer") or '').strip().upper()
        if not correct:
            # Fallback: try to parse from combined text
            m = re.search(r"Đáp\s*án\s*đúng\s*:\s*([ABCD])", cur.get("combined", ""), flags=re.IGNORECASE)
            correct = (m.group(1).upper() if m else "")

        if choice == correct:
            msg = f"Đúng rồi! Bạn mở được cửa Phòng {s['room']} → Phòng {s['room'] + 1}."
            s["room"] += 1
            s["current"] = None
            done = s["room"] > GAME_ROOMS
            if done:
                msg = "Chính xác! Bạn đã trả lời đúng 10 câu và THOÁT KHỎI MÊ CUNG TRIẾT HỌC!"

            return jsonify({
                "correct": True,
                "done": done,
                "room": min(s["room"], GAME_ROOMS),
                "rooms_total": GAME_ROOMS,
                "message": msg,
                "gatekeeper": "Tốt lắm! Cánh cửa đã mở. Hãy tiếp tục cuộc hành trình!",
            })

        # Incorrect → Gọi    AI để đưa gợi ý thông minh (KHÔNG đưa đáp án)
        question_text = cur.get("question", "")
        topic = cur.get("topic", "")
        wrong_choice = choice
        
        # Tạo prompt để chatbot đưa gợi ý
        hint_prompt = f"""Bạn là Triết học AI - trợ lý thông minh trong trò chơi "Mê Cung Triết Học". 
Người chơi vừa trả lời SAI câu hỏi sau:

Câu hỏi: {question_text}
Chủ đề: {topic}
Người chơi đã chọn: {wrong_choice}

NHIỆM VỤ: Đưa ra GỢI Ý ngắn gọn (2-3 câu) để giúp người chơi suy nghĩ đúng hướng.

QUY TẮC NGHIÊM NGẶT:
- TUYỆT ĐỐI KHÔNG được nói đáp án đúng là gì
- KHÔNG được nói "đáp án là A/B/C/D" hay "chọn đáp án..."
- Chỉ gợi ý về KIẾN THỨC, KHÁI NIỆM liên quan
- Có thể giải thích tại sao đáp án người chơi chọn chưa đúng
- Nói ngắn gọn, thân thiện, khuyến khích người chơi thử lại

Gợi ý của bạn:"""

        try:
            # Gọi chatbot AI để tạo gợi ý
            hint_response = llm.invoke(hint_prompt)
            if hasattr(hint_response, 'content'):
                ai_hint = hint_response.content
            else:
                ai_hint = str(hint_response)
            ai_hint = _sanitize_assistant_output(ai_hint)
            # Giới hạn độ dài gợi ý
            if len(ai_hint) > 300:
                ai_hint = ai_hint[:300] + "..."
        except Exception as e:
            print(f"[WARN] AI hint generation failed: {e}")
            ai_hint = f"Hãy xem lại kiến thức về {topic}. Suy nghĩ kỹ về từng đáp án nhé!"

        return jsonify({
            "correct": False,
            "done": False,
            "room": s["room"],
            "rooms_total": GAME_ROOMS,
            "message": "❌ Chưa đúng rồi. Cửa vẫn khóa — hãy thử lại!",
            "hint": ai_hint,
            "gatekeeper": ai_hint,
        })

    # Start Flask app
    port = int(getattr(args, 'port', 5002))
    host = getattr(args, 'host', '0.0.0.0')
    print(f"[backend] Flask starting on http://{host}:{port}")
    app.run(host=host, port=port, debug=getattr(args, 'debug', False), use_reloader=getattr(args, 'debug', False))

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Arguments for serve.py")

    model_group = parser.add_argument_group("Model Option")
    model_group.add_argument('-m','--mode', type=str, choices=['online', 'offline'], default='offline', help='Choose either online or offline mode system')
    model_group.add_argument('-n','--model_name', type=str, default='gemini', help='Define name of LLM model to use')
    model_group.add_argument('-e','--model_engine', type=str, default='huggingface', help='Define model engine of LLM model (Optional)')
    model_group.add_argument('-v','--model_version', type=str, default='local', help='Define model version of LLM model (Optional)')

    feature_group = parser.add_argument_group("Feature Option")
    feature_group.add_argument('--db', type=str, choices=['qdrant', 'mongodb', 'chromadb'], default='chromadb', help='Choose type of vector store database')
    feature_group.add_argument('--embedding_backend', type=str, choices=['sentence_transformers', 'fastembed'], default='sentence_transformers', help='Embedding backend: sentence_transformers (default) or fastembed (faster startup)')
    feature_group.add_argument('--embedding_model', type=str, default='BAAI/bge-m3', help='Declare what embedding model to use for RAG')
    feature_group.add_argument('--reranker', type=str, default='Alibaba-NLP/gte-multilingual-reranker-base', help='Declare name of CrossEncoder ReRanker')
    feature_group.add_argument('--enable_rerank', action='store_true', default=False, help='Enable CrossEncoder reranking (slow startup; off by default)')
    feature_group.add_argument('--enable_reflection', action='store_true', default=False, help='Enable query reflection/rewrite (adds latency; off by default)')
    feature_group.add_argument('--top_k', type=int, default=2, help='Number of retrieved passages to include (smaller=faster)')
    feature_group.add_argument('--host', type=str, default='0.0.0.0', help='Flask host bind')
    feature_group.add_argument('--port', type=int, default=5002, help='Flask port bind')
    feature_group.add_argument('--passage_max_chars', type=int, default=900, help='Max characters per retrieved passage (smaller=faster)')
    feature_group.add_argument('--max_new_tokens', type=int, default=256, help='Max tokens to generate per answer (smaller=faster)')
    feature_group.add_argument('--debug', action='store_true', default=False, help='Enable Flask debug + auto-reload (can cause brief connection drops)')

    args = parser.parse_args()
    main(args)