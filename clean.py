from pathlib import Path
import json
import shutil

BASE = Path(r"G:\Auto-thinkRAG_test\storge")  # 如路径不同请修改
DRY_RUN = False  # 先干跑，改为 False 才会删除

def main():
    to_delete = []
    for ds_path in BASE.rglob("kv_store_doc_status.json"):
        try:
            data = json.loads(ds_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] 读取失败: {ds_path} ({exc})")
            continue

        items = data.items() if isinstance(data, dict) else [
            ((item.get("doc_id") or item.get("id")), item) for item in data
        ]
        for doc_id, info in items:
            status = str(info.get("status", "")).lower()
            multi = bool(info.get("multimodal_processed", False))
            if status != "processed" or not multi:
                to_delete.append(ds_path.parent.parent)  # light -> doc root

    unique_dirs = sorted({p.resolve() for p in to_delete})
    if not unique_dirs:
        print("没有发现需要清理的目录。")
        return

    print("将删除的目录列表：")
    for p in unique_dirs:
        print(f"  {p}")

    if DRY_RUN:
        print("\nDRY_RUN=True，仅预览。修改 DRY_RUN=False 后再运行以实际删除。")
        return

    for p in unique_dirs:
        try:
            shutil.rmtree(p, ignore_errors=True)
            print(f"[DEL] {p}")
        except Exception as exc:
            print(f"[FAIL] {p} ({exc})")

if __name__ == "__main__":
    main()
