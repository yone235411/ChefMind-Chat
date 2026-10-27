import json
import os
import requests
import streamlit as st
import streamlit.components.v1 as components
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----------------------------------------------------
# 1. 页面基本配置与 Custom CSS 注入
# ----------------------------------------------------
st.set_page_config(
    page_title="ChefMind - RAG家常菜谱搭配助手",
    page_icon="🍳",
    layout="centered"
)

# 高端视觉 CSS 注入：优化字体大小、行高、气泡阴影与 Header 层次感
st.markdown("""
<style>
    /* 全局系统字体与背景优化 */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    .stApp {
        background-color: #F8FAFC;
    }

    /* 顶部 Banner 现代化设计 */
    .hero-header {
        background: linear-gradient(135deg, #FF5E62 0%, #FF9966 100%);
        padding: 28px 32px;
        border-radius: 20px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px rgba(255, 94, 98, 0.22);
    }
    .hero-title {
        font-size: 28px;
        font-weight: 800;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 15px;
        opacity: 0.95;
        margin-top: 8px;
        font-weight: 400;
        line-height: 1.4;
    }
    .badge-container {
        display: inline-flex;
        gap: 8px;
        margin-left: auto;
    }
    .badge {
        background: rgba(255, 255, 255, 0.22);
        backdrop-filter: blur(4px);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    /* 侧边栏优化 */
    [data-testid="stSidebar"] {
        background-color: #F1F5F9;
        border-right: 1px solid #E2E8F0;
    }

    /* 消息气泡字体与排版放大 */
    .stChatMessage {
        border-radius: 16px !important;
        padding: 16px 20px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03) !important;
        border: 1px solid #E2E8F0 !important;
        background-color: #FFFFFF !important;
        font-size: 15px !important;
        line-height: 1.65 !important;
    }

    /* 聊天消息中标题行加粗加大 */
    .stChatMessage h3, .stChatMessage strong {
        color: #0F172A;
        font-size: 16px;
    }

    /* RAG 参考源日志科技感文本框 */
    .stTextArea textarea {
        font-family: "Fira Code", "Courier New", monospace !important;
        font-size: 13px !important;
        background-color: #1E293B !important;
        color: #E2E8F0 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. 渲染 Header（已移除 INT6130 Demo 标签）
# ----------------------------------------------------
st.markdown("""
<div class="hero-header">
    <div class="hero-title">
        🍳 ChefMind Chat
        <div class="badge-container">
            <span class="badge">RAG Engine</span>
        </div>
    </div>
    <div class="hero-subtitle">
        基于 RAG 的家常菜谱搭配聊天机器人
    </div>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------
# 自动平滑滚动辅助函数
# ----------------------------------------------------
def auto_scroll_to_bottom():
    js_code = """
    <script>
        function scrollToBottom() {
            var mainContainer = window.parent.document.querySelector('.main');
            if (mainContainer) {
                mainContainer.scrollTop = mainContainer.scrollHeight;
            }
        }
        setTimeout(scrollToBottom, 120);
    </script>
    """
    components.html(js_code, height=0)


# ----------------------------------------------------
# 3. 侧边栏配置
# ----------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ 系统鉴权配置")
    api_key = st.text_input("🔑 API Key", type="password", placeholder="输入 DashScope / 百炼 Key")
    base_url = st.text_input(
        "🌐 Base URL Endpoint",
        value="https://ws-ribvq7y3uasfinlm.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    )

    st.divider()
    st.markdown("### 📚 私有知识库状态")
    if os.path.exists("recipes.txt"):
        st.success("🟢 已加载 `recipes.txt` (30道经典菜谱)")
    else:
        st.error("🔴 未检测到 `recipes.txt` 知识库文件")


# ----------------------------------------------------
# 4. 构建 RAG 向量数据库
# ----------------------------------------------------
@st.cache_resource(show_spinner="正在加载菜谱知识库并构建 FAISS 向量索引...")
def init_vector_store():
    loader = TextLoader("recipes.txt", encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=0,
        separators=["\n\n---\n\n", "\n---\n", "---"]
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


# ----------------------------------------------------
# 5. 对话历史渲染
# ----------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 您好！我是 ChefMind 菜谱智能管家。\n\n"
                "我能够严格基于本地菜谱私有库为您解答烹饪步骤、营养搭配或场景推荐（如：*‘推荐一道适合减脂期的高蛋白菜品’*）。请问今天想做点什么？"
            ),
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ----------------------------------------------------
# 6. RAG 问答与 API 请求逻辑
# ----------------------------------------------------
if user_input := st.chat_input("输入食材、菜名或健康需求（如：低脂、高蛋白）..."):
    if not api_key:
        st.warning("⚠️ 请先在左侧边栏配置您的 API Key 才能启动对话！")
        st.stop()

    # 展示用户提问
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    try:
        # 1. FAISS 向量检索
        vector_store = init_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        retrieved_docs = retriever.invoke(user_input)
        context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

        # 2. System Prompt 约束
        system_prompt = f"""# Role
你是一位拥有国家认证资质的营养师兼资深家常菜大厨。你的任务是基于用户提供的食材、口味偏好或营养需求，利用【检索到的菜谱知识】为用户提供精准、健康、可操作的烹饪指导与搭配建议。

# Context (检索到的知识库内容)
{context_text}

# Constraints (严格约束)
1. 【绝对真实】：你推荐的所有菜谱、食材用量、烹饪步骤，必须完全来源于上述提供的【检索到的知识库内容】。严禁凭空捏造知识库中不存在的菜品、食材或步骤！
2. 【信息缺失处理】：如果知识库中没有符合条件的菜品，请直接回答“抱歉，我的家常菜谱库中暂时没有找到符合您要求的菜品/信息”，并建议用户更换条件或尝试库内的其他菜品，绝不允许使用你自身的预训练知识来编造菜谱。
3. 【营养严谨】：如果用户提出减脂、增肌、控糖等健康需求，请严格依据知识库中提供的该菜品营养成分与禁忌进行科学解释和搭配，不要虚构数据。
4. 【步骤清晰】：输出必须条理清晰，语言通俗易懂，适合普通家庭厨房操作。
5. 【多轮指代处理】：如果用户在对话中使用“它”、“这道菜”等指代词，请结合上文指代历史对话中提到的相关菜品。

# Output Format (输出格式)
【推荐菜品】：[菜名]
【分类/标签】：[如：快捷炒菜 / 高蛋白低脂]
【所需食材】：[列出主料、辅料与具体用量]
【烹饪步骤】：
1. [步骤一]
2. [步骤二]
...
【营养与禁忌】：[依据知识库提供相关建议]
【大厨贴士】：[依据知识库中的烹饪要点给出提醒，切勿引入新食材]
"""

        with st.chat_message("assistant"):
            with st.spinner("🔍 正在检索 FAISS 向量库并进行智能推理..."):
                clean_api_key = str(api_key).strip().encode('ascii', 'ignore').decode('ascii')
                clean_base_url = str(base_url).strip().rstrip('/')

                endpoint = f"{clean_base_url}/chat/completions"

                headers = {
                    "Authorization": f"Bearer {clean_api_key}",
                    "Content-Type": "application/json; charset=utf-8"
                }

                # 构建带 3 轮历史记忆的上下文
                api_messages = [{"role": "system", "content": system_prompt}]
                recent_history = st.session_state.messages[-6:]
                for msg in recent_history:
                    api_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

                payload_dict = {
                    "model": "qwen-plus",
                    "messages": api_messages,
                    "temperature": 0.1
                }

                payload_bytes = json.dumps(payload_dict, ensure_ascii=False).encode('utf-8')
                res = requests.post(endpoint, data=payload_bytes, headers=headers, timeout=30)
                res_data = res.json()

                if res.status_code == 200:
                    ans_content = res_data["choices"][0]["message"]["content"]

                    # 渲染主回答
                    st.markdown(ans_content)

                    # 科技感 Context 数据源展示框
                    with st.expander("🧩 查看 RAG 检索日志与知识库召回依据 (Context)"):
                        st.caption("以下为从 `recipes.txt` 中检索出的 Top-4 最相似向量片段：")
                        for idx, doc in enumerate(retrieved_docs, 1):
                            lines = [line.strip() for line in doc.page_content.split('\n') if line.strip()]
                            title_line = lines[0] if lines else f"参考片段 {idx}"

                            st.markdown(f"**📍 Chunk {idx}: {title_line}**")
                            st.text_area(
                                label=f"doc_{idx}",
                                value=doc.page_content,
                                height=95,
                                disabled=True,
                                label_visibility="collapsed"
                            )

                    # 保存历史
                    st.session_state.messages.append(
                        {"role": "assistant", "content": ans_content}
                    )

                    # 自动平滑平移到底部
                    auto_scroll_to_bottom()
                else:
                    err_msg = res_data.get("error", {}).get("message", res.text)
                    st.error(f"API 调用失败 ({res.status_code}): {err_msg}")

    except Exception as e:
        st.error(f"发生错误: {str(e)}")