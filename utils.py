# utils.py
import os
import json
from typing import List, Dict, Optional
from openai import OpenAI


# ---------------- LLM 调用封装（通义千问 OpenAI-compatible） ----------------

# 使用通义千问（Dashscope 兼容 OpenAI）
client = OpenAI(
    api_key="sk-6a99dd24b9924c62a236f3543aebb4ef",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def call_llm(messages, model: str = "qwen-plus", temperature: float = 0.3) -> str:
    """
    封装对通义千问的调用。
    如果网络或证书出错，直接把异常信息返回到前端，方便调试。
    """
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return completion.choices[0].message.content
    except Exception as e:
        # 不抛异常，直接把错误内容当字符串返回，避免整个推演中断
        return f"[LLM 调用出错：{repr(e)}]"


# ========== 简单知识库 & RAG ==========

def load_text_files_from_folder(folder_path: str) -> List[str]:
    """
    从某个目录递归读取文本文件作为知识库内容。
    这里只示例：.txt / .md / .log 等纯文本。
    如果需要 docx / pdf，自行扩展。
    """
    docs: List[str] = []
    if not folder_path:
        return docs

    folder_path = os.path.expanduser(folder_path)
    if not os.path.isdir(folder_path):
        return docs

    exts = {".txt", ".md", ".log"}
    for root, _, files in os.walk(folder_path):
        for fname in files:
            _, ext = os.path.splitext(fname)
            if ext.lower() not in exts:
                continue
            full_path = os.path.join(root, fname)
            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read().strip()
                    if text:
                        docs.append(f"【文件：{fname}】\n{text}")
            except Exception as e:
                docs.append(f"【读取失败：{full_path}】{e}")
    return docs


def simple_keyword_retrieval(docs: List[str], query: str, top_k: int = 3) -> List[str]:
    """
    极简 RAG：按关键字重叠数打分，然后取前 top_k。
    方便 Demo，避免引入额外依赖（向量模型等）。
    """
    if not docs or not query:
        return []

    # 简单按空格切词即可，实际使用可替换为 jieba 分词 + TF-IDF / 向量召回等
    q_tokens = set(query.split())
    scored = []

    for d in docs:
        score = 0
        for t in q_tokens:
            if t and t in d:
                score += 1
        if score > 0:
            scored.append((score, d))

    if not scored:
        # 没匹配到就随便给几个
        return docs[:top_k]

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]
