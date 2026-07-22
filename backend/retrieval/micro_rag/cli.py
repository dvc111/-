from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import Triple
from .retriever import MicroRetriever
from .training import train_from_jsonl
from .contracts import build_micro_evidence_subgraph


def _retrieve(args: argparse.Namespace) -> None:
    record = json.loads(Path(args.input).read_text(encoding="utf-8"))
    retriever = MicroRetriever(rounds=args.rounds, model_path=args.model)
    if "macro_subgraph" in record:
        result = build_micro_evidence_subgraph(
            record,
            retriever=retriever,
            top_k=args.top_k,
            threshold=args.threshold,
            require_mlp=args.require_mlp,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    triples = [Triple.from_dict(value, index) for index, value in enumerate(record["triples"])]
    result = retriever.retrieve(
        question=record["question"],
        topic_entities=record["topic_entities"],
        triples=triples,
        top_k=args.top_k,
        threshold=args.threshold,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


def _train(args: argparse.Namespace) -> None:
    metrics = train_from_jsonl(args.data, args.output, args.rounds, args.epochs)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="SubgraphRAG 微观检索模块")
    subparsers = parser.add_subparsers(dest="command", required=True)

    retrieve_parser = subparsers.add_parser("retrieve", help="筛选 Top-K 微观证据")
    retrieve_parser.add_argument("--input", required=True, help="宏观子图 JSON")
    retrieve_parser.add_argument("--model", help="可选的已训练 MLP 参数")
    retrieve_parser.add_argument("--top-k", type=int, default=5)
    retrieve_parser.add_argument("--threshold", type=float, default=0.0)
    retrieve_parser.add_argument("--rounds", type=int, default=2)
    retrieve_parser.add_argument(
        "--require-mlp",
        action="store_true",
        help="正式模式：未通过 --model 加载 MLP 时直接报错",
    )
    retrieve_parser.set_defaults(func=_retrieve)

    train_parser = subparsers.add_parser("train", help="用最短路径弱监督训练 MLP")
    train_parser.add_argument("--data", required=True, help="JSONL 训练文件")
    train_parser.add_argument("--output", required=True, help="模型输出路径")
    train_parser.add_argument("--epochs", type=int, default=300)
    train_parser.add_argument("--rounds", type=int, default=2)
    train_parser.set_defaults(func=_train)

    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as error:
        parser.exit(2, f"错误：{error}\n")


if __name__ == "__main__":
    main()
