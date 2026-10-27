"""项目根入口，方便直接 `streamlit run app.py`。

真正的界面在 src/zhisaotong_agent/app.py。这里只做一件事：
把 src/ 加进模块搜索路径，这样不用先 pip install -e . 也能跑起来。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from zhisaotong_agent.app import main  # noqa: E402

main()
