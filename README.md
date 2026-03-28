# CodeAdvisor

CodeAdvisor 是一个基于 LangChain 和 LangGraph 的智能代码分析工具。它通过 RAG (检索增强生成) 技术对项目代码进行扫描，并使用多级 LLM 工作流（Summarizer 总结 + Advisor 建议）为开发者提供深度、结构化的代码见解。

## 🌟 核心特性

- **多模型协作工作流**: 使用 LangGraph 构建 `代码总结 -> 改进建议` 的串行分析链。
- **智能 RAG 系统**: 
  - 支持 **Chroma** (持久化) 和 **FAISS** (内存) 向量库。
  - 针对 Windows 环境优化的路径扫描与文件加载。
  - 严格的数据清洗，移除空字节与非法元数据，适配多种 API 提供商。
- **广泛的模型兼容性**:
  - 原生支持 **OpenAI**, **DeepSeek**, **DashScope** (阿里云) 等兼容 OpenAI 协议的接口。
  - 支持为 Summarizer、Advisor 和 Embeddings 分别配置不同的 API Key 和 Base URL。
- **交互式 CLI 配置**: 友好的向导式界面，支持一键保存配置，支持增量或全量重新索引。
- **多语言支持**: 默认支持 Python (`.py`)，可扩展支持 JS/TS、C++、Java 等。

## 📂 项目结构

```text
CodeAdvisor/
├── main.py              # 入口文件，负责启动交互配置与分析流程
├── src/
│   ├── interface.py     # 交互式 CLI 引导逻辑
│   ├── workflow.py      # LangGraph 工作流定义 (Summarizer -> Advisor)
│   ├── rag.py           # RAG 核心逻辑 (加载、分片、索引、检索)
│   ├── models.py        # 模型初始化工厂 (LLM & Embeddings)
│   └── configls.py      # 配置管理 (YAML 读写)
├── config/
│   ├── dir.config.yaml  # 持久化配置文件
│   └── .env             # 环境变量 (API Keys)
├── data/
│   └── chroma/          # 默认持久化向量存储目录
└── output/              # 默认分析结果输出目录
```

## 🚀 快速开始

### 1. 安装依赖

推荐使用 Python 3.12+。

```bash
pip install -r requirements.txt
# 或者如果你使用的是 pyproject.toml
pip install .
```

### 2. 配置环境变量

在项目根目录或 `config/` 目录下创建 `.env` 文件：

```env
# 基础 OpenAI Key (可选，作为兜底)
OPENAI_API_KEY=sk-xxxx

# 专门用于总结模型的 Key (OPENAI_API_KEY1)
OPENAI_API_KEY1=sk-xxxx

# 专门用于建议模型的 Key (OPENAI_API_KEY2)
OPENAI_API_KEY2=sk-xxxx

# 专门用于向量模型的 Key (OPENAI_API_KEY_EMBED)
OPENAI_API_KEY_EMBED=sk-xxxx
```

### 3. 运行分析

```bash
python main.py
```

根据终端提示，完成以下配置：
1. 指定待分析的项目路径。
2. 设置输出路径。
3. 选择是否需要重新索引（首次运行请选 `y`）。
4. 选择模型（如 `deepseek-chat`, `gpt-4`）及其 Base URL。

## 🛠️ 技术细节

- **数据清洗**: 为了解决 DashScope 等提供商对输入格式的严格要求，我们在 [rag.py](src/rag.py) 中实现了极致的清洗逻辑，确保所有发送至 Embedding 接口的文本不含空字节 (`\x00`) 且非空。
- **批处理优化**: 针对 DashScope 兼容模式，我们在 [models.py](src/models.py) 中强制设置 `chunk_size=1` 并关闭本地上下文长度校验，以确保 API 调用稳定性。
- **配置持久化**: 所有选择都会自动保存至 `config/dir.config.yaml`，下次运行时可直接回车跳过重复输入。

## 📝 输出结果

分析完成后，结果将保存在 `output/` 目录下：
- `summary.md`: 代码库的整体架构、核心模块及潜在问题的总结。
- `suggestions.md`: 针对代码质量、性能、安全性和维护性的具体改进建议。

---
*Powered by LangChain & LangGraph*
