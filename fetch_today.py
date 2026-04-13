#!/usr/bin/env python3
"""
Daily-GasTurbineML: fetch_today.py
检索燃气轮机 + PINN / 深度学习领域最新论文

Usage / 用法:
    python fetch_today.py                    # 默认每类10篇（全量刷新）
    python fetch_today.py --per_category 15  # 每类15篇
    python fetch_today.py --append 10        # 向每类追加10篇新论文（不覆盖现有）
    python fetch_today.py --no_code_search   # 跳过代码搜索（更快）
    python fetch_today.py --token YOUR_TOKEN # 使用 GitHub token
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_gasturbine_papers

# Delegate entirely to gen_gasturbine_papers.main()
if __name__ == "__main__":
    gen_gasturbine_papers.main()
