# -*- coding: utf-8 -*-
"""
语义切片策略
- 基于句子边界进行切分
- 使用连接词与主题重叠度保持语义完整性
- 支持句子级重叠，避免上下文割裂
"""

import re
from typing import List, Set


def split_into_sentences(text: str) -> List[str]:
    """按中英句子结束符切分，保留结束符与紧随的右引号/括号。"""
    if not text:
        return []

    enders = set("。！？!?；;")
    right_closers = set('”’"\')）】』》〉]')
    sentences: List[str] = []
    buffer: List[str] = []

    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        buffer.append(ch)

        if ch in enders:
            j = i + 1
            while j < n and text[j] in right_closers:
                buffer.append(text[j])
                j += 1
            sentences.append("".join(buffer).strip())
            buffer = []
            i = j
            continue
        i += 1

    tail = "".join(buffer).strip()
    if tail:
        sentences.append(tail)

    # 合并过短的句子（如标题/断行）
    merged: List[str] = []
    for s in sentences:
        if merged and len(s) < 8:
            merged[-1] = (merged[-1] + (" " if not merged[-1].endswith(("。", "！", "？", ".", "!", "?")) else "") + s).strip()
        else:
            merged.append(s)
    return [s for s in merged if s]


CN_CONNECTIVES = [
    "但是", "然而", "因此", "此外", "另外", "同时", "而且", "并且", "而", "所以", "因为", "例如", "比如",
    "总之", "首先", "其次", "最后", "另一方面", "综上", "基于此", "换言之", "也就是说", "即", "即使",
]
EN_CONNECTIVES = [
    "however", "therefore", "moreover", "in addition", "furthermore", "besides",
    "meanwhile", "thus", "hence", "so", "because", "for example", "for instance",
    "in other words", "that is", "i.e.", "e.g.",
]
STRIP_PREFIX = '“”"\'（()【[〈《『「'  # 句首可剥离的引号/括号


def starts_with_connective(sentence: str) -> bool:
    s = sentence.strip()
    s = s.lstrip(STRIP_PREFIX).strip().lower()
    for p in CN_CONNECTIVES:
        if s.startswith(p.lower()):
            return True
    for p in EN_CONNECTIVES:
        if s.startswith(p.lower()):
            return True
    return False


CN_STOPWORDS: Set[str] = {
    "的", "了", "和", "及", "与", "或", "而", "并", "在", "对", "把", "被", "是", "就", "也", "都", "很", "更", "还",
    "则", "即", "等", "及其", "以及", "其中", "通过", "对于", "关于", "由于", "因此", "此外", "另外",
}
EN_STOPWORDS: Set[str] = {
    "the", "a", "an", "of", "to", "and", "or", "but", "for", "on", "in", "at",
    "is", "are", "was", "were", "be", "been", "being", "by", "with", "as",
    "that", "this", "these", "those", "it", "its", "from",
}


def extract_keywords(text: str) -> Set[str]:
    """
    极简关键词提取：
    - 英文：按单词提取，去停用词，长度≥2
    - 中文：提取连续汉字串，保留长度≥2
    """
    tokens: Set[str] = set()

    # 英文单词
    for w in re.findall(r"[A-Za-z]+", text):
        lw = w.lower()
        if len(lw) >= 2 and lw not in EN_STOPWORDS:
            tokens.add(lw)

    # 连续中文串
    for seq in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(seq) >= 2:
            tokens.add(seq)

    # 数字串（如日期/金额）
    for num in re.findall(r"\d{2,}", text):
        tokens.add(num)

    # 简单归一化：常见符号去掉
    normalized = set()
    for t in tokens:
        t2 = re.sub(r"[，。,.\-_/\\:：；;、\s]+", "", t)
        if t2:
            # 过长中文串截取为前4-8长度，减少过拟合
            if re.fullmatch(r"[\u4e00-\u9fff]+", t2) and len(t2) > 8:
                normalized.add(t2[:8])
            else:
                normalized.add(t2)
    return normalized


def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def semantic_chunking(
    text: str,
    target_chunk_size: int = 500,
    min_chunk_size: int = 350,
    max_chunk_size: int = 800,
    overlap_sentences: int = 1,
    similarity_threshold: float = 0.25,
    tail_compare_k: int = 3,
) -> List[str]:
    """
    语义切片：
    - 先按句子切分
    - 按目标大小聚合句子
    - 结合连接词与主题重叠度决定是否与下一句合并
    - 达到上限强制截断
    - 句子级重叠
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current_sentences: List[str] = []

    i = 0
    n = len(sentences)

    while i < n:
        next_sentence = sentences[i].strip()
        if not next_sentence:
            i += 1
            continue

        current_text = " ".join(current_sentences).strip()
        next_len = len(next_sentence)
        current_len = len(current_text)

        # 尽量填充到 target_chunk_size
        if current_len + next_len <= target_chunk_size:
            current_sentences.append(next_sentence)
            i += 1
            continue

        # 未达最小阈值时，允许继续追加
        if current_len < min_chunk_size:
            current_sentences.append(next_sentence)
            i += 1
            # 如果超出 max，则马上截断
            if len(" ".join(current_sentences)) >= max_chunk_size:
                chunk_text = " ".join(current_sentences).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                # 构造重叠
                overlap_tail = current_sentences[-overlap_sentences:] if overlap_sentences > 0 else []
                current_sentences = overlap_tail[:]
            continue

        # 语义判断：连接词或主题重叠 → 倾向继续合并
        should_merge = False
        if starts_with_connective(next_sentence):
            should_merge = True
        else:
            tail_scope = " ".join(current_sentences[-tail_compare_k:])
            sim = jaccard_similarity(extract_keywords(tail_scope), extract_keywords(next_sentence))
            if sim >= similarity_threshold:
                should_merge = True

        if should_merge and (current_len + next_len) <= max_chunk_size:
            current_sentences.append(next_sentence)
            i += 1
            continue

        # 截断并处理重叠
        chunk_text = " ".join(current_sentences).strip()
        if chunk_text:
            chunks.append(chunk_text)

        overlap_tail = current_sentences[-overlap_sentences:] if overlap_sentences > 0 else []
        current_sentences = overlap_tail[:]

        # 如果当前句过长导致无法合并，单独成块（防止死循环/极长句）
        if not current_sentences and next_len >= max_chunk_size:
            chunks.append(next_sentence)
            i += 1

    # 收尾
    tail_text = " ".join(current_sentences).strip()
    if tail_text:
        chunks.append(tail_text)

    return chunks


def print_chunk_analysis(chunks: List[str], method_name: str) -> None:
    """打印切片分析结果"""
    print(f"\n{'='*60}")
    print(f"📋 {method_name}")
    print(f"{'='*60}")

    if not chunks:
        print("❌ 未生成任何切片")
        return

    total_length = sum(len(chunk) for chunk in chunks)
    avg_length = total_length / len(chunks)
    min_length = min(len(chunk) for chunk in chunks)
    max_length = max(len(chunk) for chunk in chunks)

    print(f"📊 统计信息:")
    print(f"   - 切片数量: {len(chunks)}")
    print(f"   - 平均长度: {avg_length:.1f} 字符")
    print(f"   - 最短长度: {min_length} 字符")
    print(f"   - 最长长度: {max_length} 字符")
    print(f"   - 长度方差: {max_length - min_length} 字符")

    print(f"\n📝 切片内容:")
    for i, chunk in enumerate(chunks, 1):
        print(f"   块 {i} ({len(chunk)} 字符):")
        print(f"   {chunk}")
        print()


if __name__ == "__main__":
    print("语义切片策略测试")
    text = """
IT之家 7 月 31 日消息，阶跃星辰宣布新一代基础大模型 Step 3 正式开源，Step 3 API 已上线阶跃星辰开放平台（platform.stepfun.com），用户也可以在“阶跃 AI”官网（stepfun.com）和“阶跃 AI”App 进行体验。

据介绍，Step 3 的多模态能力围绕“轻量视觉路径”与“稳定协同训练”展开，重点解决视觉引入带来的 token 负担与训练干扰问题。为此，其采用 5B Vision Encoder，并通过双层 2D 卷积对视觉特征进行降采样，将视觉 token 数量减少到原来的 1/16，减轻上下文长度压力，提升推理效率。

IT之家附官方对 Step 3 模型的介绍如下：

核心要点
Step 3 兼顾智能与效率，专为追求性能与成本极致均衡的企业和开发者设计，旨在面向推理时代打造最适合应用的模型。

Step 3 采用 MoE 架构，总参数量 321B，激活参数量 38B。

Step 3 拥有强大的视觉感知和复杂推理能力，可准确完成跨领域的复杂知识理解、数学与视觉信息的交叉分析，以及日常生活中的各类视觉分析问题。

通过 MFA（Multi-matrix Factorization Attention） & AFD（Attention-FFN Disaggregation）的优化，在各类芯片上推理效率均大幅提升。

面向 AFD 场景的 StepMesh 通信库已随模型一同开源，提供可跨硬件的标准部署接口，支持关键性能在实际服务中的稳定复现。

模型限时折扣中，所有请求均按最低价格计算，每百万 token 价格低至输入 1.5 元，输出 4 元。

Step 3 API 已上线阶跃星辰开放平台（platform.stepfun.com），大家也可以在“阶跃 AI”官网（stepfun.com）和“阶跃 AI”App（应用商店搜索下载）进行体验。
""".strip()

    print(f"测试文本长度: {len(text)} 字符")

    chunks = semantic_chunking(
        text,
        target_chunk_size=500,
        min_chunk_size=350,
        max_chunk_size=800,
        overlap_sentences=1,
        similarity_threshold=0.25,
        tail_compare_k=3,
    )
    print_chunk_analysis(chunks, "语义切片（基于句子边界 + 主题重叠 + 连接词）")
