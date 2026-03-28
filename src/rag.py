# src/rag.py

import os
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.parsers.language import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
from src.models import get_embeddings


# =========================
# 🌍 多语言映射
# =========================

LANGUAGE_MAP = {
    ".py": Language.PYTHON,
    # ".js": Language.JS,
    # ".ts": Language.TS,
    # ".cpp": Language.CPP,
    # ".c": Language.CPP,
    # ".java": Language.JAVA,
    # ".go": Language.GO,
}


# =========================
# 📌 IGNORE 处理
# =========================

def should_ignore(path: Path, ignore_list: List[str]) -> bool:
    """Checks if a path should be ignored based on the ignore list."""
    path_str = str(path.as_posix())
    for ignore in ignore_list:
        if not ignore:
            continue
        if f"/{ignore}/" in f"/{path_str}/" or path_str.startswith(f"{ignore}/"):
            return True
    return False


# =========================
# 📌 多语言加载
# =========================

def load_and_split_code(root_dir: str, config: Dict[str, Any]) -> List[Document]:
    root_path = Path(root_dir).resolve()
    print(f"📂 正在扫描目录: {root_path}")
    
    ignore_list = config.get("ignore_dir", [])
    print(f"🚫 忽略列表: {ignore_list}")

    all_chunks = []
    
    for suffix, lang in LANGUAGE_MAP.items():
        print(f"🔍 搜索 {suffix} 文件...")
        files = []
        try:
            for p in root_path.rglob(f"*{suffix}"):
                relative_p = p.relative_to(root_path)
                if p.is_file() and not should_ignore(relative_p, ignore_list):
                    files.append(p)
        except Exception as e:
            print(f"⚠️ 扫描目录时出错: {e}")
            continue

        if not files:
            print(f"ℹ️ 未发现任何 {suffix} 文件。")
            continue

        print(f"🛠️ 发现 {len(files)} 个 {suffix} 文件，正在解析内容...")
        
        for i, file_path in enumerate(files):
            try:
                loader = TextLoader(str(file_path), encoding='utf-8')
                file_docs = loader.load()
                
                for d in file_docs:
                    d.metadata["source"] = str(file_path.relative_to(root_path))
                
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=lang,
                    chunk_size=config.get("chunk_size", 2000),
                    chunk_overlap=config.get("chunk_overlap", 200),
                )
                chunks = splitter.split_documents(file_docs)
                
                # PRE-FILTER: Only keep documents with substantial text and clean it
                for chunk in chunks:
                    if chunk.page_content:
                        clean_content = str(chunk.page_content).replace("\x00", "").strip()
                        if clean_content:
                            chunk.page_content = clean_content
                            all_chunks.append(chunk)
                
                if (i + 1) % 10 == 0 or (i + 1) == len(files):
                    print(f"  进度: {i+1}/{len(files)} 文件已处理...")
                    
            except Exception as e:
                print(f"  ❌ 无法处理文件 {file_path}: {e}")

        print(f"✅ {suffix} 处理完毕: {len(files)} 个文件 → 当前总代码块: {len(all_chunks)}")

    if not all_chunks:
        print("⚠️ 警告: 未发现任何符合条件的代码文件。请检查 input_dir 和 ignore_dir 配置。")
    
    return all_chunks


# =========================
# 📦 向量库（支持 FAISS / Chroma）
# =========================

def create_vectorstore(texts: List[Document], config: Dict[str, Any]):
    if not texts:
        raise ValueError("No documents to index")

    db_type = config.get("vector_db", "chroma")
    persist_dir = config.get("persist_dir", "./data/chroma")
    # Use centralized embedding initialization
    embeddings = get_embeddings(
        model_name=config.get("model_embed"), 
        api_key_env="OPENAI_API_KEY_EMBED", 
        base_url=config.get("base_url_embed")
    )

    # FINAL CLEANING: Strictly ensure no malformed documents reach the API
    final_docs = []
    for doc in texts:
        if hasattr(doc, 'page_content') and isinstance(doc.page_content, str):
            # Remove null bytes and other weird stuff that crashes some JSON parsers
            clean_content = doc.page_content.replace("\x00", "").strip()
            if clean_content:
                doc.page_content = clean_content
                # Also clean metadata values to be strings/ints only
                clean_metadata = {}
                for k, v in doc.metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        clean_metadata[k] = v
                    else:
                        clean_metadata[k] = str(v)
                doc.metadata = clean_metadata
                final_docs.append(doc)
    
    if not final_docs:
        raise ValueError("没有有效的非空文本可用于构建向量库。")

    print(f"🔄 准备索引 {len(final_docs)} 条代码块...")

    if db_type == "faiss":
        print(f"⚡ FAISS 初始化...")
        return FAISS.from_documents(final_docs, embeddings)

    elif db_type == "chroma":
        print(f"📦 Chroma 初始化 (持久化: {persist_dir})...")
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        return Chroma.from_documents(
            final_docs,
            embedding=embeddings,
            persist_directory=persist_dir
        )
    else:
        raise ValueError(f"Unsupported vector DB: {db_type}")


def load_vectorstore(config: Dict[str, Any]):
    """Loads an existing vector store from disk."""
    db_type = config.get("vector_db", "chroma")
    persist_dir = config.get("persist_dir", "./data/chroma")

    # Use centralized embedding initialization
    embeddings = get_embeddings(
        model_name=config.get("model_embed"), 
        api_key_env="OPENAI_API_KEY_EMBED", 
        base_url=config.get("base_url_embed")
    )
    
    if db_type == "chroma":
        if not Path(persist_dir).exists():
            raise FileNotFoundError(f"向量库目录不存在: {persist_dir}")
        print(f"📦 正在加载现有 Chroma 向量库: {persist_dir}")
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        raise ValueError(f"当前模式不支持直接加载 {db_type}。")


# =========================
# 🔎 Retriever
# =========================

def create_retriever(vectorstore, config: Dict[str, Any]):
    k = config.get("top_k", 4)
    return vectorstore.as_retriever(search_kwargs={"k": k})


# =========================
# 🚀 一体化接口（给 workflow 用）
# =========================

def build_retriever(input_dir: str, config: Dict[str, Any]):
    reindex = config.get("reindex", True)
    
    if reindex:
        texts = load_and_split_code(input_dir, config)
        if not texts:
            raise ValueError("没有加载到任何文档")
        vectorstore = create_vectorstore(texts, config)
        print("✨ 向量库索引构建完成。")
    else:
        vectorstore = load_vectorstore(config)
        print("✨ 向量库加载成功。")
    
    retriever = create_retriever(vectorstore, config)
    return retriever
