from dataclasses import dataclass, asdict
import torch
import argparse
from pathlib import Path
from deepforest import main
from config import load_config
import pandas as pd
import numpy as np
import json

# Load configuration
config = load_config()
model_config = config["model"]
data_config = config["data"]
pred_config = config["prediction"]

evaluation_config = config["evaluation"]


@dataclass(frozen=True)
class EvaluationResult:
    model_name: str
    avg_iou: float
    box_recall: float
    box_precision: float
    
    
def save_results(evaluation_result: EvaluationResult, dir: Path):
    content = json.dumps(asdict(evaluation_result))
    dest_path = dir / f"{evaluation_result.model_name}.json"
    
    with open(dest_path, "w") as file:
        file.write(content) 

def evaluate(model_path: Path | None):
    """Load a fine-tuned model and evaluate it on a test set."""
    if model_path and not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
    test_annotations_path = Path(evaluation_config["annotations"])
    if not test_annotations_path.exists():
        print(f"Error: Test annotations file not found at {test_annotations_path}")
        return

    model = main.deepforest()
    if model_path:
        print(f"Loading fine-tuned model from {model_path}...")
        model.model = torch.load(model_path, weights_only=False)
        model.config["score_thresh"] = evaluation_config["score_thresh"]
    else:
        print("Loading the pretrained model")
        model_path = "weecology/deepforest-tree"
        model.load_model(model_name="weecology/deepforest-tree", revision="main")

    print(f"Evaluating model {model_path} on test data from {test_annotations_path}...")
    
    # The root_dir should point to the directory where the images for evaluation are.
    # Based on your config, it is data_config["processed_data_dir"]
    results = model.evaluate(
        csv_file=str(test_annotations_path),
        root_dir=evaluation_config["data"]
    )

    print("\nEvaluation Results:")
    print("-------------------")
    
    # The results object from evaluate is a dictionary containing a pandas dataframe
    # See https://deepforest.readthedocs.io/en/latest/evaluate.html
    if "results" in results and not results["results"].empty:
        print(results["results"])
        results_df: pd.DataFrame = results["results"]
        results_df.to_csv(Path("evalutation_out.csv"))
        
        everything = [str(s) for s in results.keys()]
        
        with open(Path("tutto.txt"), "w") as txt_file:
            txt_file.write(",".join(everything))
    else:
        print("No evaluation results were generated. The test set might be empty or paths might be incorrect.")

    if "box_recall" in results and "box_precision" in results:
        print(f"\nBox Recall: {results['box_recall']:.3f}")
        print(f"Box Precision: {results['box_precision']:.3f}")
    else:
        print("\nCould not calculate box recall and precision.")
        
    er = EvaluationResult(
        model_name=str(model_path).split('/')[-1].split(".")[0] if model_path else "weecology_deepforest-tree",
        avg_iou=np.average(results_df["IoU"].to_numpy()),
        box_recall = results['box_recall'],
        box_precision = results['box_precision'],
    )
    
    save_results(er, Path("data/03_results"))
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model.")
    parser.add_argument("--model_path", required=False, default=None, type=Path, help="Path to the fine-tuned model file (.pt).")
    # parser.add_argument("--test_annotations", required=True, type=Path, help="Path to the CSV file with test annotations.")
    args = parser.parse_args()
    
    evaluate(args.model_path)#, args.test_annotations)
