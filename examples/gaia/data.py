import os
import re
import string
import warnings
from pathlib import Path
from typing import Any, Iterator, Union

import huggingface_hub
from datasets import load_dataset
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

DATASET_CACHE_DIR = Path(__file__).parent / "gaia_data"


class BenchmarkTask(BaseModel):
    task_id: Union[str, list[str]]
    question: Union[str, list[str]]
    file_path: Union[str, list[str]]
    file_name: Union[str, list[str]]
    ground_truth: Union[str, list[str]]
    difficulty: Union[str, list[str]]
    metadata: dict[Any, Any] = {}


class GaiaBenchmark:
    def __init__(self, benchmark_settings):
        self.benchmark_settings = benchmark_settings
        self._data = None

    def download(self):
        difficulty = self.benchmark_settings["difficulty"]
        if difficulty == "all":
            config = "2023_all"
        else:
            config = f"2023_level{difficulty}"
        split = self.benchmark_settings["split"]

        self._data = load_dataset(
            "gaia-benchmark/GAIA", config, split=split, trust_remote_code=True, token=HF_TOKEN
        )

        huggingface_hub.snapshot_download(
            "gaia-benchmark/GAIA",
            repo_type="dataset",
            local_dir=str(DATASET_CACHE_DIR),
            token=HF_TOKEN,
        )

    def _accuracy(self, responses: list[str], gts: list[str]):
        correct = 0
        total = len(responses)

        for i, r in enumerate(responses):
            if self.is_correct_answer(r, gts[i]):
                correct += 1

        accuracy = correct / total if total > 0 else 0.0
        return accuracy

    def compute_metrics(self, responses: list[str], gts: list[str]):
        acc = self._accuracy(responses, gts)
        return {"accuracy": acc}

    def _get_task(self, item):
        file_path = item["file_path"]
        if file_path:
            absolute_file_path = (Path(DATASET_CACHE_DIR) / file_path).resolve()
            file_path = str(absolute_file_path)

        return BenchmarkTask(
            task_id=item["task_id"],
            question=item["Question"],
            file_path=file_path,
            file_name=item["file_name"],
            ground_truth=item["Final answer"],
            difficulty=item["Level"],
            metadata=item["Annotator Metadata"],
        )

    def __getitem__(self, idx):
        item = self._data[idx]  # type: ignore
        return self._get_task(item)

    def __iter__(self) -> Iterator[BenchmarkTask]:
        for item in self._data:  # type: ignore
            yield self._get_task(item)

    def __len__(self):
        return len(self._data)  # type: ignore

    def normalize_number_str(self, number_str: str) -> float:
        for char in ["$", "%", ","]:
            number_str = number_str.replace(char, "")
        try:
            return float(number_str)
        except ValueError:
            return float("inf")

    def split_string(
        self,
        s: str,
        char_list: list[str] = [",", ";"],
    ) -> list[str]:
        pattern = f"[{''.join(char_list)}]"
        return re.split(pattern, s)

    def is_correct_answer(
        self,
        model_answer: str,
        ground_truth: str,
    ) -> bool:
        def is_float(element: any) -> bool:  # type: ignore
            try:
                float(element)
                return True
            except ValueError:
                return False

        if is_float(ground_truth):
            normalized_answer = self.normalize_number_str(model_answer)
            return normalized_answer == float(ground_truth)

        elif any(char in ground_truth for char in [",", ";"]):
            gt_elems = self.split_string(ground_truth)
            ma_elems = self.split_string(model_answer)

            if len(gt_elems) != len(ma_elems):
                warnings.warn("Answer lists have different lengths, returning False.", UserWarning)
                return False

            comparisons = []
            for ma_elem, gt_elem in zip(ma_elems, gt_elems):
                if is_float(gt_elem):
                    normalized_ma_elem = self.normalize_number_str(ma_elem)
                    comparisons.append(normalized_ma_elem == float(gt_elem))
                else:
                    comparisons.append(
                        self.normalize_str(ma_elem, remove_punct=False)
                        == self.normalize_str(gt_elem, remove_punct=False)
                    )
            return all(comparisons)

        else:
            return self.normalize_str(model_answer) == self.normalize_str(ground_truth)

    def normalize_str(self, input_str, remove_punct=True) -> str:
        no_spaces = re.sub(r"\s", "", input_str)

        if remove_punct:
            translator = str.maketrans("", "", string.punctuation)
            return no_spaces.lower().translate(translator)
        else:
            return no_spaces.lower()
