#!/usr/bin/env python3
"""
Daily-GasTurbineML: Gas Turbine + Physics-Informed Neural Network / Deep Learning
每日文献追踪系统 — 燃气轮机与叶轮机械机器学习

6 categories:
  1. PINN for Gas Turbines & Turbomachinery
  2. Surrogate Models for Gas Turbine Performance
  3. Deep Learning for Turbomachinery Aerodynamics
  4. CFD-ML for Turbomachinery Flows
  5. ML for Combustion & Thermal Analysis
  6. Gas Turbine Health Monitoring & Fault Diagnosis
"""
import os, re, time, datetime, argparse
import arxiv, requests

TOKEN = os.environ.get("GITHUB_TOKEN", "")
OWNER = "xgxgnpu"
REPO_NAME = "Daily-GasTurbineML"
BASE = os.path.dirname(os.path.abspath(__file__))
GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"

# ─── Category Definitions ───────────────────────────────────────────

CATEGORIES = {
    "pinn_gasturbine": {
        "name_cn": "PINN — 燃气轮机与叶轮机械",
        "name_en": "PINN for Gas Turbines & Turbomachinery",
        "query": '(ti:"physics-informed" OR ti:"PINN" OR abs:"physics-informed neural network") AND (abs:"gas turbine" OR abs:"turbomachinery" OR abs:"compressor" OR abs:"turbine blade" OR abs:"turbine cascade" OR abs:"axial turbine" OR abs:"centrifugal compressor" OR abs:"turbojet" OR abs:"turbo")',
        "filter": "pinn_gasturbine",
        "per_category": 10,
    },
    "surrogate_performance": {
        "name_cn": "代理模型 — 性能预测",
        "name_en": "Surrogate Models for Gas Turbine Performance",
        "query": '(abs:"surrogate model" OR abs:"surrogate" OR abs:"neural network" OR abs:"deep learning" OR abs:"machine learning") AND (abs:"gas turbine" OR abs:"turbine performance" OR abs:"compressor performance" OR abs:"turbomachinery performance" OR abs:"engine performance" OR abs:"turbine efficiency" OR abs:"compressor map")',
        "filter": "surrogate_performance",
        "per_category": 10,
    },
    "dl_aerodynamics": {
        "name_cn": "深度学习 — 叶轮机内流与气动",
        "name_en": "Deep Learning for Turbomachinery Aerodynamics",
        "query": '(abs:"deep learning" OR abs:"neural network" OR abs:"convolutional neural" OR abs:"transformer" OR abs:"graph neural") AND (abs:"turbomachinery" OR abs:"turbine blade" OR abs:"compressor blade" OR abs:"rotor" OR abs:"stator" OR abs:"airfoil" OR abs:"blade passage" OR abs:"cascade flow")',
        "filter": "dl_aerodynamics",
        "per_category": 10,
    },
    "cfd_ml": {
        "name_cn": "CFD 与机器学习融合",
        "name_en": "CFD-ML for Turbomachinery Flows",
        "query": '(abs:"machine learning" OR abs:"deep learning" OR abs:"neural network" OR abs:"data-driven") AND (abs:"turbomachinery" OR abs:"gas turbine" OR abs:"compressor" OR abs:"turbine") AND (abs:"CFD" OR abs:"computational fluid dynamics" OR abs:"flow simulation" OR abs:"RANS" OR abs:"turbulence model" OR abs:"flow field reconstruction")',
        "filter": "cfd_ml",
        "per_category": 10,
    },
    "combustion_ml": {
        "name_cn": "燃烧室机器学习",
        "name_en": "ML for Combustion & Thermal Analysis",
        "query": '(abs:"machine learning" OR abs:"deep learning" OR abs:"neural network" OR abs:"physics-informed") AND (abs:"combustion" OR abs:"gas turbine combustor" OR abs:"combustion chamber" OR abs:"flame" OR abs:"ignition" OR abs:"NOx" OR abs:"emissions" OR abs:"thermal" OR abs:"heat transfer" OR abs:"cooling") AND (abs:"gas turbine" OR abs:"turbine" OR abs:"engine")',
        "filter": "combustion_ml",
        "per_category": 10,
    },
    "health_monitoring": {
        "name_cn": "健康监测与故障诊断",
        "name_en": "Gas Turbine Health Monitoring & Fault Diagnosis",
        "query": '(abs:"deep learning" OR abs:"machine learning" OR abs:"neural network") AND (abs:"gas turbine" OR abs:"turbine engine" OR abs:"turbomachinery" OR abs:"aeroengine" OR abs:"jet engine") AND (abs:"fault diagnosis" OR abs:"anomaly detection" OR abs:"health monitoring" OR abs:"remaining useful life" OR abs:"degradation" OR abs:"prognostics" OR abs:"vibration")',
        "filter": "health_monitoring",
        "per_category": 10,
    },
}

# ─── Keyword pools ──────────────────────────────────────────────────

GASTURBINE_KEYWORDS = [
    "gas turbine", "turbomachinery", "compressor", "turbine blade",
    "axial turbine", "centrifugal compressor", "turbojet", "turbofan",
    "turbine cascade", "rotor", "stator", "blade passage",
]

PINN_KEYWORDS = [
    "PINN", "physics-informed neural", "physics informed neural",
    "physics-informed deep", "physics-informed machine",
]

ML_KEYWORDS = [
    "machine learning", "deep learning", "neural network",
    "convolutional neural", "transformer", "LSTM", "GAN",
    "surrogate model", "data-driven", "reinforcement learning",
    "graph neural", "encoder", "decoder",
]

AERO_KEYWORDS = [
    "turbomachinery", "turbine blade", "compressor blade", "rotor",
    "stator", "airfoil", "cascade flow", "blade passage",
    "impeller", "diffuser", "nozzle", "vane",
]

CFD_KEYWORDS = [
    "CFD", "computational fluid dynamics", "RANS", "LES", "DNS",
    "flow simulation", "turbulence model", "flow field",
    "Navier-Stokes", "Reynolds-averaged",
]

COMBUSTION_KEYWORDS = [
    "combustion", "combustor", "flame", "ignition", "NOx", "emissions",
    "heat transfer", "cooling", "thermal", "thermodynamics",
]

HEALTH_KEYWORDS = [
    "fault diagnosis", "anomaly detection", "health monitoring",
    "remaining useful life", "degradation", "prognostics",
    "vibration", "bearing", "rotating machinery", "condition monitoring",
]

KW_POOL = [
    "PINN", "Physics-Informed", "Gas Turbine", "Turbomachinery",
    "Compressor", "Turbine Blade", "Deep Learning", "Surrogate Model",
    "Neural Operator", "CFD", "RANS", "Combustion", "Health Monitoring",
    "Fault Diagnosis", "Aerodynamics", "Performance Prediction",
    "Data-Driven", "Neural Network", "Transfer Learning", "Optimization",
    "Flow Field", "Heat Transfer", "Remaining Useful Life", "Anomaly Detection",
]


def has_any(text, keywords):
    tl = text.lower()
    return any(k.lower() in tl for k in keywords)


def filter_paper(filter_type, title, abstract):
    text = title + " " + abstract
    if filter_type == "pinn_gasturbine":
        return has_any(text, PINN_KEYWORDS) and has_any(text, GASTURBINE_KEYWORDS)
    elif filter_type == "surrogate_performance":
        return has_any(text, ML_KEYWORDS) and has_any(text, GASTURBINE_KEYWORDS)
    elif filter_type == "dl_aerodynamics":
        return has_any(text, ML_KEYWORDS) and has_any(text, AERO_KEYWORDS)
    elif filter_type == "cfd_ml":
        return has_any(text, ML_KEYWORDS) and has_any(text, CFD_KEYWORDS) and has_any(text, GASTURBINE_KEYWORDS + AERO_KEYWORDS)
    elif filter_type == "combustion_ml":
        return has_any(text, ML_KEYWORDS) and has_any(text, COMBUSTION_KEYWORDS) and has_any(text, GASTURBINE_KEYWORDS + ["turbine", "engine"])
    elif filter_type == "health_monitoring":
        return has_any(text, ML_KEYWORDS) and has_any(text, HEALTH_KEYWORDS) and has_any(text, GASTURBINE_KEYWORDS + ["rotating machinery", "aeroengine", "jet engine"])
    return False


def extract_keywords(title, abstract):
    combined = title + " " + abstract
    found = []
    for kw in KW_POOL:
        if kw.lower() in combined.lower() and kw not in found:
            found.append(kw)
        if len(found) >= 5:
            break
    return ", ".join(found) if found else "Gas Turbine, Neural Network"


def search_code(arxiv_id, title):
    if not TOKEN:
        return None
    headers = {"Authorization": f"token {TOKEN}"}
    for q in [f"arxiv:{arxiv_id}", title[:80]]:
        try:
            resp = requests.get(
                GITHUB_SEARCH_URL,
                params={"q": q, "sort": "stars", "per_page": 3},
                headers=headers, timeout=10,
            )
            if resp.status_code == 200:
                false_patterns = ["awesome", "daily", "arxiv-daily", "paper-list", "survey"]
                for item in resp.json().get("items", []):
                    if not any(p in item["full_name"].lower() for p in false_patterns):
                        return item["html_url"]
        except Exception:
            pass
        time.sleep(0.5)
    return None


# ─── Fetch papers ────────────────────────────────────────────────────

def fetch_category(cat_key, cat_info, existing_ids, max_results=10):
    print(f"\n  [{cat_info['name_en']}] searching arXiv...")
    client = arxiv.Client()
    search = arxiv.Search(
        query=cat_info["query"],
        max_results=max_results * 8,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    papers = []
    try:
        results = list(client.results(search))
    except Exception as e:
        print(f"  arXiv error: {e}")
        return papers

    for r in results:
        if len(papers) >= max_results:
            break
        aid = r.get_short_id().split("v")[0]
        if aid in existing_ids:
            continue
        title = r.title.replace("\n", " ").strip()
        abstract = r.summary.replace("\n", " ").strip()

        if not filter_paper(cat_info["filter"], title, abstract):
            print(f"  SKIP (filter): {title[:60]}...")
            continue

        authors = ", ".join([a.name for a in r.authors[:5]])
        if len(r.authors) > 5:
            authors += " et al."
        date_str = r.published.strftime("%Y-%m-%d")
        keywords = extract_keywords(title, abstract)
        code_link = search_code(aid, title)

        papers.append({
            "title": title,
            "authors": authors,
            "date": date_str,
            "keywords": keywords,
            "arxiv_id": aid,
            "arxiv_link": f"https://arxiv.org/abs/{aid}",
            "pdf_link": f"https://arxiv.org/pdf/{aid}",
            "code": code_link if code_link else "无",
            "abstract": abstract,
        })
        existing_ids.add(aid)
        print(f"  NEW [{len(papers)}/{max_results}] {date_str} | {title[:60]}...")

    # Re-fetch full abstracts by ID (batch search may truncate)
    if papers:
        ids = [p["arxiv_id"] for p in papers]
        try:
            full_results = list(arxiv.Client().results(arxiv.Search(id_list=ids)))
            abstract_map = {
                r.get_short_id().split("v")[0]: r.summary.replace("\n", " ").strip()
                for r in full_results
            }
            for p in papers:
                if p["arxiv_id"] in abstract_map:
                    p["abstract"] = abstract_map[p["arxiv_id"]]
            print(f"  Re-fetched full abstracts for {len(abstract_map)} papers")
        except Exception as e:
            print(f"  Warning: re-fetch abstracts failed: {e}")

    print(f"  => {len(papers)} papers found")
    return papers


# ─── Parse existing MD ───────────────────────────────────────────────

def parse_papers_by_category_from_md(md_path):
    """Parse existing papers grouped by category from GasTurbineML MD file."""
    if not os.path.exists(md_path):
        return {k: [] for k in CATEGORIES}
    with open(md_path, encoding="utf-8") as f:
        content = f.read()

    name_to_key = {info["name_cn"]: key for key, info in CATEGORIES.items()}

    cat_ids_ordered = {k: [] for k in CATEGORIES}
    list_match = re.search(r'## 论文列表\n(.*?)(?=\n## 详细摘要|\Z)', content, re.DOTALL)
    if list_match:
        list_section = '\n' + list_match.group(1)
        parts = re.split(r'\n### (.+?)（\d+ 篇）\n', list_section)
        for i in range(1, len(parts), 2):
            cat_name_cn = parts[i].strip()
            cat_content = parts[i + 1] if i + 1 < len(parts) else ""
            cat_key = name_to_key.get(cat_name_cn)
            if cat_key is None:
                continue
            ids = re.findall(r'\[([0-9.]+)\]\(https://arxiv\.org/abs/[0-9.]+\)', cat_content)
            cat_ids_ordered[cat_key] = ids

    paper_by_id = {}
    detail_match = re.search(r'## 详细摘要\n+(.*)', content, re.DOTALL)
    if detail_match:
        detail_content = '\n' + detail_match.group(1)
        blocks = re.split(r'\n### \d+\. ', detail_content)
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            lines_b = block.split('\n')
            title = lines_b[0].strip()
            if not title:
                continue
            authors_m = re.search(r'\*\*作者\*\*:\s*(.+)', block)
            date_m = re.search(r'\*\*发表日期\*\*:\s*(\S+)', block)
            kw_m = re.search(r'\*\*关键词\*\*:\s*(.+)', block)
            link_m = re.search(r'\*\*论文链接\*\*:\s*\[([0-9.]+)\]\((https://arxiv\.org/abs/[^)]+)\)\s*\|\s*\[PDF\]\(([^)]+)\)', block)
            code_m = re.search(r'\*\*代码\*\*:\s*\[Code\]\(([^)]+)\)', block)
            abstract_m = re.search(r'>\s*(.+?)(?:\n---|\Z)', block, re.DOTALL)
            if not (date_m and link_m):
                continue
            arxiv_id = link_m.group(1)
            code_val = code_m.group(1) if code_m else "无"
            paper_by_id[arxiv_id] = {
                "title": title,
                "authors": authors_m.group(1).strip() if authors_m else "",
                "date": date_m.group(1).strip(),
                "keywords": kw_m.group(1).strip() if kw_m else "Gas Turbine, Neural Network",
                "arxiv_id": arxiv_id,
                "arxiv_link": link_m.group(2),
                "pdf_link": link_m.group(3),
                "code": code_val,
                "abstract": abstract_m.group(1).strip().rstrip('-').strip() if abstract_m else "",
            }

    result = {}
    for cat_key in CATEGORIES:
        ids = cat_ids_ordered.get(cat_key, [])
        result[cat_key] = [paper_by_id[aid] for aid in ids if aid in paper_by_id]
    total_parsed = sum(len(v) for v in result.values())
    print(f"  Parsed {total_parsed} existing papers from {os.path.basename(md_path)}")
    return result


# ─── Write markdown ──────────────────────────────────────────────────

def write_md_cn(all_papers_by_cat, filepath):
    total = sum(len(ps) for ps in all_papers_by_cat.values())
    today = datetime.date.today().strftime("%Y.%m.%d")
    lines = [
        f"# 燃气轮机机器学习 — 每日文献追踪（PINN / 深度学习 / 叶轮机械）\n\n",
        f"> 更新日期: {today}\n",
        f"> 检索关键词: 燃气轮机, 叶轮机械, PINN, 深度学习, 代理模型, CFD, 燃烧, 故障诊断\n",
        f"> 共 {total} 篇论文，按{len(CATEGORIES)}个类别分类\n\n",
        f"## 论文列表\n\n",
    ]
    for cat_key, papers in all_papers_by_cat.items():
        cat_info = CATEGORIES[cat_key]
        lines.append(f"### {cat_info['name_cn']}（{len(papers)} 篇）\n\n")
        lines.append("| # | 发表日期 | 标题 | 作者 | 关键词 | 论文链接 | 代码 |\n")
        lines.append("|:---:|:--------:|:-----|:-----|:-------|:-------:|:----:|\n")
        for i, p in enumerate(papers):
            code_str = f"[Code]({p['code']})" if p['code'] != '无' else '无'
            first_author = p['authors'].split(',')[0].strip()
            lines.append(
                f"| {i+1} | {p['date']} | **{p['title']}** | {first_author} et al. "
                f"| {p['keywords']} | [{p['arxiv_id']}]({p['arxiv_link']}) | {code_str} |\n"
            )
        lines.append("\n")
    lines.append("## 详细摘要\n\n")
    idx = 0
    for cat_key, papers in all_papers_by_cat.items():
        for p in papers:
            idx += 1
            code_line = f"[Code]({p['code']})" if p['code'] != '无' else '无'
            lines.append(f"### {idx}. {p['title']}\n\n")
            lines.append(f"- **作者**: {p['authors']}\n")
            lines.append(f"- **发表日期**: {p['date']}\n")
            lines.append(f"- **关键词**: {p['keywords']}\n")
            lines.append(f"- **论文链接**: [{p['arxiv_id']}]({p['arxiv_link']}) | [PDF]({p['pdf_link']})\n")
            lines.append(f"- **代码**: {code_line}\n\n")
            lines.append(f"> {p['abstract']}\n---\n\n")
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"  Written {total} papers to {os.path.basename(filepath)}")


def write_md_en(all_papers_by_cat, filepath):
    total = sum(len(ps) for ps in all_papers_by_cat.values())
    today = datetime.date.today().strftime("%Y.%m.%d")
    lines = [
        f"# Gas Turbine Machine Learning — Daily Paper Tracker\n\n",
        f"> Updated: {today}\n",
        f"> Keywords: Gas Turbine, Turbomachinery, PINN, Deep Learning, Surrogate Model, CFD, Combustion, Fault Diagnosis\n",
        f"> Total: {total} papers across {len(CATEGORIES)} categories\n\n",
        f"## Paper List\n\n",
    ]
    for cat_key, papers in all_papers_by_cat.items():
        cat_info = CATEGORIES[cat_key]
        lines.append(f"### {cat_info['name_en']} ({len(papers)} papers)\n\n")
        lines.append("| # | Date | Title | Authors | Keywords | Paper | Code |\n")
        lines.append("|:---:|:--------:|:-----|:-----|:-------|:-------:|:----:|\n")
        for i, p in enumerate(papers):
            code_str = f"[Code]({p['code']})" if p['code'] != '无' else 'N/A'
            first_author = p['authors'].split(',')[0].strip()
            lines.append(
                f"| {i+1} | {p['date']} | **{p['title']}** | {first_author} et al. "
                f"| {p['keywords']} | [{p['arxiv_id']}]({p['arxiv_link']}) | {code_str} |\n"
            )
        lines.append("\n")
    lines.append("## Detailed Abstracts\n\n")
    idx = 0
    for cat_key, papers in all_papers_by_cat.items():
        for p in papers:
            idx += 1
            code_line = f"[Code]({p['code']})" if p['code'] != '无' else 'N/A'
            lines.append(f"### {idx}. {p['title']}\n\n")
            lines.append(f"- **Authors**: {p['authors']}\n")
            lines.append(f"- **Date**: {p['date']}\n")
            lines.append(f"- **Keywords**: {p['keywords']}\n")
            lines.append(f"- **Paper**: [{p['arxiv_id']}]({p['arxiv_link']}) | [PDF]({p['pdf_link']})\n")
            lines.append(f"- **Code**: {code_line}\n\n")
            lines.append(f"> {p['abstract']}\n---\n\n")
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"  Written {total} papers to {os.path.basename(filepath)}")


# ─── README generators ───────────────────────────────────────────────

AUTHOR_BLOCK_EN = """## Author

**Xiong Xiong (熊雄)**

- Northwestern Polytechnical University (NWPU)
- Research interests: AI4PDE, Physics-Informed Deep Learning, Data-Driven Discovery
- Email: xiongxiongnwpu@mail.nwpu.edu.cn
- [Google Scholar](https://scholar.google.com/citations?user=j1M9tkwAAAAJ&hl=zh-CN&oi=sra)
- [ResearchGate](https://www.researchgate.net/profile/Xiong-Xiong-19?ev=hdr_xprf)
- [Physics-informed-vibe-coding](https://github.com/xgxgnpu/Physics-informed-vibe-coding)

---
"""

AUTHOR_BLOCK_CN = """## 作者

**熊雄 (Xiong Xiong)**

- 西北工业大学 (NWPU)
- 研究方向：AI4PDE、物理信息深度学习、数据驱动发现
- 邮箱：xiongxiongnwpu@mail.nwpu.edu.cn
- [Google Scholar](https://scholar.google.com/citations?user=j1M9tkwAAAAJ&hl=zh-CN&oi=sra)
- [ResearchGate](https://www.researchgate.net/profile/Xiong-Xiong-19?ev=hdr_xprf)
- [Physics-informed-vibe-coding](https://github.com/xgxgnpu/Physics-informed-vibe-coding) — 首个专注于 PINN Vibe Coding 研究的开源仓库。

---
"""


def gen_readme_en(total):
    return f"""# Daily-GasTurbineML

> Daily tracking of Physics-Informed Neural Networks & Deep Learning papers for Gas Turbines and Turbomachinery.

[中文版](README_CN.md)

{AUTHOR_BLOCK_EN}

## Overview

This repository independently tracks the latest research on **machine learning and physics-informed methods applied to gas turbines and turbomachinery**. It covers **{total} papers** across 6 categories.

Topics include: PINN for turbomachinery flows, surrogate models for performance prediction, deep learning for aerodynamic design, CFD-ML hybrid methods, combustion ML, and gas turbine health monitoring.

> **Quick Navigation:**
>
> [1. PINN-GasTurbine](#1-pinn-for-gas-turbines--turbomachinery) | [2. Surrogate](#2-surrogate-models-for-gas-turbine-performance) | [3. DL-Aero](#3-deep-learning-for-turbomachinery-aerodynamics) | [4. CFD-ML](#4-cfd-ml-for-turbomachinery-flows) | [5. Combustion](#5-ml-for-combustion--thermal-analysis) | [6. Health](#6-gas-turbine-health-monitoring--fault-diagnosis)

---

### 1. PINN for Gas Turbines & Turbomachinery

Physics-Informed Neural Networks applied directly to gas turbine and turbomachinery simulations:

- PINN for compressor and turbine blade aerodynamics
- Physics-constrained flow field prediction in blade passages
- Inverse design and parameter identification in turbomachinery
- Multi-fidelity PINN combining CFD data with physical laws

### 2. Surrogate Models for Gas Turbine Performance

Data-driven surrogate models replacing expensive simulations for performance prediction:

- Neural network surrogate for gas turbine cycle analysis
- Multi-point compressor/turbine map modeling
- Gaussian process and deep learning for off-design performance
- Reduced-order modeling for engine simulation acceleration

### 3. Deep Learning for Turbomachinery Aerodynamics

Deep learning applied to aerodynamic analysis and design in turbomachinery:

- CNN/Transformer for blade flow field prediction
- Generative models for turbine blade shape optimization
- Graph neural networks for cascade aerodynamic analysis
- Attention-based models for wake interaction and unsteady flows

### 4. CFD-ML for Turbomachinery Flows

Machine learning augmenting or replacing CFD in turbomachinery:

- ML-enhanced turbulence closure models (RANS, LES)
- Physics-informed data assimilation for turbomachinery CFD
- Neural operators for fast flow field prediction
- ML-corrected RANS for secondary flows in blade passages

### 5. ML for Combustion & Thermal Analysis

Machine learning for combustion chambers and thermal management in gas turbines:

- Neural network combustion models for ignition and flame dynamics
- ML-based NOx and emission prediction
- Deep learning for turbine cooling design and optimization
- Thermal field reconstruction from sparse sensor data

### 6. Gas Turbine Health Monitoring & Fault Diagnosis

Deep learning for condition monitoring, fault detection and prognostics in gas turbines:

- LSTM/CNN for gas turbine degradation prediction
- Anomaly detection in vibration and temperature sensor data
- Remaining useful life estimation for turbine components
- Transfer learning for cross-engine fault diagnosis

---

**Total papers**: {total}

## Usage

```bash
pip install arxiv requests

# Fetch latest papers (10 per category)
python fetch_today.py

# Specify number per category
python fetch_today.py --per_category 15

# Append N new papers per category without overwriting
python fetch_today.py --append 10

# Skip code search for faster execution
python fetch_today.py --no_code_search
```

## License

MIT
"""


def gen_readme_cn(total):
    return f"""# Daily-GasTurbineML

> 燃气轮机 + 物理信息神经网络 / 深度学习领域每日文献追踪。

[English Version](README.md)

{AUTHOR_BLOCK_CN}

## 概述

本仓库独立追踪 **机器学习与物理信息方法用于燃气轮机及叶轮机械** 的最新研究，涵盖6个类别，共 **{total} 篇论文**。

> **快速导航** — 点击下方任意类别跳转：
>
> [1. PINN-燃气轮机](#1-pinn--燃气轮机与叶轮机械) | [2. 代理模型](#2-代理模型--性能预测) | [3. 深度学习-气动](#3-深度学习--叶轮机内流与气动) | [4. CFD-ML](#4-cfd-与机器学习融合) | [5. 燃烧](#5-燃烧室机器学习) | [6. 健康监测](#6-健康监测与故障诊断)

---

### 1. PINN — 燃气轮机与叶轮机械

物理信息神经网络直接应用于燃气轮机与叶轮机械仿真：

- 压气机/涡轮叶片气动的PINN方法
- 叶片流道中基于物理约束的流场预测
- 叶轮机械中的反问题与参数识别
- 融合CFD数据与物理定律的多保真度PINN

### 2. 代理模型 — 性能预测

替代昂贵仿真的数据驱动代理模型：

- 燃气轮机循环分析的神经网络代理
- 多工况压气机/涡轮特性图建模
- 偏设计点性能的高斯过程和深度学习方法
- 面向发动机仿真加速的降阶模型

### 3. 深度学习 — 叶轮机内流与气动

深度学习用于叶轮机械气动分析与设计：

- CNN/Transformer用于叶片流场预测
- 生成模型用于涡轮叶片形状优化
- 图神经网络用于叶栅气动分析
- 基于注意力机制的尾迹干涉和非定常流动

### 4. CFD 与机器学习融合

机器学习增强或替代叶轮机械CFD：

- ML增强湍流封闭模型（RANS、LES）
- 叶轮机械CFD的物理信息数据同化
- 快速流场预测的神经算子
- 叶片流道二次流的ML修正RANS

### 5. 燃烧室机器学习

燃气轮机燃烧室与热管理的机器学习：

- 点火和火焰动力学的神经网络燃烧模型
- 基于ML的NOx与排放预测
- 涡轮冷却设计与优化的深度学习方法
- 稀疏传感器数据的温度场重建

### 6. 健康监测与故障诊断

深度学习用于燃气轮机状态监测、故障检测和预测性维护：

- LSTM/CNN用于燃气轮机退化预测
- 振动和温度传感器数据的异常检测
- 涡轮部件剩余使用寿命估计
- 跨发动机故障诊断的迁移学习

---

**论文总数**: {total}

## 使用方法

```bash
pip install arxiv requests

# 检索最新论文（每类10篇）
python fetch_today.py

# 指定每类数量
python fetch_today.py --per_category 15

# 追加模式（不覆盖现有）
python fetch_today.py --append 10

# 跳过代码搜索（更快）
python fetch_today.py --no_code_search
```

## 许可

MIT
"""


# ─── HTML generator ──────────────────────────────────────────────────

def gen_index_html(all_papers_by_cat, total):

    def make_category_html(papers, cat_info, start_idx, lang):
        na = "无" if lang == "cn" else "N/A"
        label = {
            "authors": "作者" if lang == "cn" else "Authors",
            "date": "发表日期" if lang == "cn" else "Date",
            "kw": "关键词" if lang == "cn" else "Keywords",
            "link": "论文链接" if lang == "cn" else "Paper",
            "code": "代码" if lang == "cn" else "Code",
            "papers": "篇" if lang == "cn" else "papers",
        }
        cat_name = cat_info["name_cn"] if lang == "cn" else cat_info["name_en"]
        html = f'<h2>{cat_name}（{len(papers)} {label["papers"]}）</h2>\n'
        th = (f'<th>#</th><th>{label["date"]}</th><th>Title</th>'
              f'<th>{label["authors"]}</th><th>{label["kw"]}</th>'
              f'<th>{label["link"]}</th><th>{label["code"]}</th>')
        html += f'<table><thead><tr>{th}</tr></thead><tbody>\n'
        for i, p in enumerate(papers):
            code_td = f'<a href="{p["code"]}" target="_blank">Code</a>' if p["code"] != "无" else na
            arxiv_td = f'<a href="{p["arxiv_link"]}" target="_blank">{p["arxiv_id"]}</a>'
            html += (f'<tr><td>{start_idx+i+1}</td><td>{p["date"]}</td>'
                     f'<td><strong>{p["title"]}</strong></td>'
                     f'<td>{p["authors"].split(",")[0].strip()} et al.</td>'
                     f'<td>{p["keywords"]}</td>'
                     f'<td>{arxiv_td}</td><td>{code_td}</td></tr>\n')
        html += '</tbody></table>\n'
        for i, p in enumerate(papers):
            code_str = f'<a href="{p["code"]}" target="_blank">Code</a>' if p["code"] != "无" else na
            arxiv_html = f'<a href="{p["arxiv_link"]}" target="_blank">{p["arxiv_id"]}</a>'
            pdf_html = f' | <a href="{p["pdf_link"]}" target="_blank">PDF</a>'
            html += f'''<div class="paper-detail">
<h3>{start_idx+i+1}. {p["title"]}</h3>
<ul>
<li><strong>{label["authors"]}</strong>: {p["authors"]}</li>
<li><strong>{label["date"]}</strong>: {p["date"]}</li>
<li><strong>{label["kw"]}</strong>: {p["keywords"]}</li>
<li><strong>{label["link"]}</strong>: {arxiv_html}{pdf_html}</li>
<li><strong>{label["code"]}</strong>: {code_str}</li>
</ul>
<blockquote>{p["abstract"][:600]}{"..." if len(p["abstract"]) > 600 else ""}</blockquote>
</div>\n'''
        return html

    cn_sections = ""
    en_sections = ""
    idx = 0
    for cat_key, papers in all_papers_by_cat.items():
        cn_sections += make_category_html(papers, CATEGORIES[cat_key], idx, "cn")
        en_sections += make_category_html(papers, CATEGORIES[cat_key], idx, "en")
        idx += len(papers)

    css = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.6; color: #24292f; background: #fff;
  max-width: 1200px; margin: 0 auto; padding: 20px;
}
@media (prefers-color-scheme: dark) {
  body { color: #c9d1d9; background: #0d1117; }
  a { color: #58a6ff; }
  table { border-color: #30363d; }
  th { background: #161b22; }
  tr:nth-child(even) { background: #161b22; }
  .tab-btn { background: #21262d; color: #c9d1d9; border-color: #30363d; }
  .tab-btn.active { background: #1f6feb; color: #fff; border-color: #1f6feb; }
  blockquote { background: #161b22; border-color: #30363d; }
}
h1 { font-size: 1.8em; margin-bottom: 8px; }
h2 { margin-top: 28px; margin-bottom: 12px; border-bottom: 2px solid #d0d7de; padding-bottom: 6px; }
.subtitle { color: #656d76; margin-bottom: 20px; font-size: 0.95em; }
.tabs { display: flex; gap: 8px; margin: 20px 0; border-bottom: 2px solid #d0d7de; }
.tab-btn {
  padding: 8px 24px; border: 1px solid #d0d7de; border-bottom: none;
  border-radius: 6px 6px 0 0; background: #f6f8fa; cursor: pointer;
  font-size: 14px; font-weight: 600; transition: all 0.2s;
}
.tab-btn.active { background: #0969da; color: #fff; border-color: #0969da; }
.tab-btn:hover:not(.active) { background: #eaeef2; }
#section-cn, #section-en { display: none; }
#section-cn.active, #section-en.active { display: block; }
table { width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 0.9em; }
th, td { padding: 10px 12px; border: 1px solid #d0d7de; text-align: left; }
th { background: #f6f8fa; font-weight: 600; white-space: nowrap; }
tr:nth-child(even) { background: #f6f8fa; }
blockquote {
  padding: 12px 16px; margin: 8px 0 16px; background: #f6f8fa;
  border-left: 4px solid #0969da; border-radius: 0 6px 6px 0;
  font-size: 0.9em; color: #656d76;
}
.paper-detail { margin: 20px 0; padding-bottom: 16px; border-bottom: 1px solid #d0d7de; }
.paper-detail h3 { margin-bottom: 8px; color: #0969da; }
.paper-detail ul { list-style: none; padding: 0; }
.paper-detail li { margin: 4px 0; }
a { color: #0969da; text-decoration: none; }
a:hover { text-decoration: underline; }
.meta { display: flex; gap: 20px; flex-wrap: wrap; margin: 12px 0; font-size: 0.9em; color: #656d76; }
.footer {
  margin-top: 40px; padding-top: 16px; border-top: 1px solid #d0d7de;
  text-align: center; font-size: 0.85em; color: #656d76;
}
"""

    js = """
function switchLang(lang) {
  document.getElementById('section-cn').style.display = (lang === 'cn') ? 'block' : 'none';
  document.getElementById('section-en').style.display = (lang === 'en') ? 'block' : 'none';
  document.getElementById('btn-cn').className = (lang === 'cn') ? 'tab-btn active' : 'tab-btn';
  document.getElementById('btn-en').className = (lang === 'en') ? 'tab-btn active' : 'tab-btn';
}
"""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Daily-GasTurbineML</title>
<style>{css}</style>
</head>
<body>

<h1>Daily-GasTurbineML</h1>
<div class="meta">
<span>GitHub: <a href="https://github.com/{OWNER}/{REPO_NAME}">{OWNER}/{REPO_NAME}</a></span>
<span>Updated: {datetime.date.today()}</span>
<span>Total: {total} papers</span>
</div>

<div class="tabs">
  <button id="btn-cn" class="tab-btn active" onclick="switchLang('cn')">中文</button>
  <button id="btn-en" class="tab-btn" onclick="switchLang('en')">English</button>
</div>

<div id="section-cn" class="active">
<h2>燃气轮机机器学习 — 每日文献追踪</h2>
<p class="subtitle">共 {total} 篇论文，{len(CATEGORIES)}个类别 | 追踪燃气轮机与叶轮机械领域 PINN / 深度学习最新进展</p>
{cn_sections}
</div>

<div id="section-en">
<h2>Gas Turbine Machine Learning — Daily Paper Tracker</h2>
<p class="subtitle">Total: {total} papers, {len(CATEGORIES)} categories | Tracking PINN & Deep Learning for Gas Turbines and Turbomachinery</p>
{en_sections}
</div>

<div class="footer">
  <p>Auto-generated by <a href="https://github.com/{OWNER}/{REPO_NAME}">Daily-GasTurbineML</a> |
  Data from <a href="https://arxiv.org" target="_blank">arXiv</a></p>
</div>

<script>{js}</script>
</body>
</html>"""


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Daily-GasTurbineML paper fetcher")
    parser.add_argument("--per_category", type=int, default=10)
    parser.add_argument("--append", type=int, default=0,
                        help="Append N new papers per category (no overwrite)")
    parser.add_argument("--no_code_search", action="store_true")
    parser.add_argument("--token", type=str, default="")
    args = parser.parse_args()

    if args.token:
        global TOKEN
        TOKEN = args.token

    md_path = os.path.join(BASE, "GasTurbineML_today.md")

    if args.append > 0:
        print(f"\n[Append mode] Reading existing papers from {md_path}...")
        existing_by_cat = parse_papers_by_category_from_md(md_path)
        all_existing_ids = set()
        for papers in existing_by_cat.values():
            for p in papers:
                all_existing_ids.add(p["arxiv_id"])
        print(f"  Global existing IDs: {len(all_existing_ids)}")

        all_papers = {}
        for cat_key, cat_info in CATEGORIES.items():
            print(f"\n=== {cat_info['name_en']} (APPEND +{args.append}) ===")
            new_papers = fetch_category(cat_key, cat_info, all_existing_ids, args.append)
            new_papers.sort(key=lambda x: x["date"], reverse=True)
            existing_cat = existing_by_cat.get(cat_key, [])
            all_papers[cat_key] = existing_cat + new_papers
            print(f"  Category total: {len(existing_cat)} + {len(new_papers)} = {len(all_papers[cat_key])}")
    else:
        existing_ids = set()
        if os.path.exists(md_path):
            with open(md_path) as f:
                existing_ids = set(re.findall(r"arxiv\.org/abs/([0-9.]+)", f.read()))
            print(f"Loaded {len(existing_ids)} existing IDs")

        all_papers = {}
        for cat_key, cat_info in CATEGORIES.items():
            print(f"\n=== {cat_info['name_en']} ===")
            papers = fetch_category(cat_key, cat_info, existing_ids, args.per_category)
            papers.sort(key=lambda x: x["date"], reverse=True)
            all_papers[cat_key] = papers

    total = sum(len(ps) for ps in all_papers.values())
    print(f"\nTotal: {total} papers")

    write_md_cn(all_papers, os.path.join(BASE, "GasTurbineML_today.md"))
    write_md_en(all_papers, os.path.join(BASE, "GasTurbineML_today_en.md"))

    html = gen_index_html(all_papers, total)
    with open(os.path.join(BASE, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print("  Written index.html")

    with open(os.path.join(BASE, "README.md"), "w", encoding="utf-8") as f:
        f.write(gen_readme_en(total))
    with open(os.path.join(BASE, "README_CN.md"), "w", encoding="utf-8") as f:
        f.write(gen_readme_cn(total))
    print("  Written READMEs")
    print(f"\nAll done! {total} papers in {BASE}")


if __name__ == "__main__":
    main()
