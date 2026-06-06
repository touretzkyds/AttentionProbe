import json
import numpy as np
import matplotlib.pyplot as plt
import os

from utils import ModelManager

class DuplicateHeadScorer:
    def __init__(self, model_name="google/flan-t5-large"):
        print("loading model")
        self.model = ModelManager(model_name)
        self.model.load_model()

        self.num_layers = self.model.config.num_layers
        self.num_heads = self.model.config.num_heads

        print(f"model loaded")

    def load_dataset(self, dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    def get_tokens_and_attentions(self, sentence):
        outputs = self.model.get_attention_outputs([sentence])

        inputs = self.model.tokenizer(
            [sentence],
            padding=True,
            return_tensors="pt"
        )

        tokens = self.model.tokenizer.convert_ids_to_tokens(
            inputs.input_ids[0]
        )

        return tokens, outputs.encoder_attentions

    def find_duplicate_positions(self, tokens, target_name):
        cleaned = [
            t.replace("▁", "")
            for t in tokens
        ]

        target = target_name.lower()

        positions = []

        for i in range(len(cleaned)):

            token = cleaned[i].lower()

            if token == target:
                positions.append(i)
                continue

            if i < len(cleaned) - 1:

                merged = (
                    cleaned[i] +
                    cleaned[i + 1]
                ).lower()

                if merged == target:
                    positions.append(i)

        return positions
    
    def collect_scatter_data(self, dataset_path, layer, head):

        dataset = self.load_dataset(dataset_path)

        x_ab, y_ab = [], []
        x_ba, y_ba = [], []

        used = 0
        skipped = 0

        for idx, item in enumerate(dataset):

            probe = item["sentence"]
            control = item["control_sentence"]

            duplicate_name = item["probe_name"]
            replacement_name = item["control_name"]

            tokens_p, att_p_all = self.get_tokens_and_attentions(probe)
            tokens_c, att_c_all = self.get_tokens_and_attentions(control)

            dup_probe = self.find_duplicate_positions(
                tokens_p,
                duplicate_name
            )

            dup_control = self.find_duplicate_positions(
                tokens_c,
                duplicate_name
            )

            repl_control = self.find_duplicate_positions(
                tokens_c,
                replacement_name
            )

            if (
                len(dup_probe) != 2
                or len(dup_control) != 1
                or len(repl_control) != 1
            ):
                skipped += 1

                print(f"\n[SKIP] Example {idx}")
                print("Duplicate:", duplicate_name)
                print("Replacement:", replacement_name)
                print("dup_probe =", dup_probe)
                print("dup_control =", dup_control)
                print("repl_control =", repl_control)

                continue

            i_p, j_p = dup_probe

            i_c = dup_control[0]
            j_c = repl_control[0]

            att_p = (
                att_p_all[layer][0, head]
                .detach()
                .cpu()
                .numpy()
            )

            att_c = (
                att_c_all[layer][0, head]
                .detach()
                .cpu()
                .numpy()
            )

            probe_ab = att_p[i_p, j_p]
            control_ab = att_c[i_c, j_c]

            x_ab.append(probe_ab)
            y_ab.append(control_ab)

            probe_ba = att_p[j_p, i_p]
            control_ba = att_c[j_c, i_c]

            x_ba.append(probe_ba)
            y_ba.append(control_ba)

            used += 1

        print("used pairs:", used)
        print("skipped pairs:", skipped)

        return x_ab, y_ab, x_ba, y_ba
    def plot_scatter(self, x_ab, y_ab, x_ba, y_ba, layer, head):

        if len(x_ab) == 0:
            print("no data points found")
            return

        plt.figure(figsize=(6, 6))

        plt.scatter(x_ab, y_ab, marker="o", alpha=0.7, label="A → B")
        plt.scatter(x_ba, y_ba, marker="+", alpha=0.7, label="B → A")

        all_vals = x_ab + y_ab + x_ba + y_ba
        mn, mx = min(all_vals), max(all_vals)

        plt.plot([mn, mx], [mn, mx], "--", color="black", alpha=0.5)

        plt.xlabel("Probe (duplicate sentence)")
        plt.ylabel("Control (non-duplicate sentence)")
        plt.title(f"Layer {layer}, Head {head}")

        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        os.makedirs("scatter_plots", exist_ok=True)
        output_path = f"scatter_plots/scatter_layer{layer}_head{head}.png"

        plt.savefig(output_path, dpi=300, bbox_inches="tight")

        print(f"saved plot to: {output_path}")

        plt.show()

if __name__ == "__main__":
    scorer = DuplicateHeadScorer()

    layer = 2
    head = 6

    dataset_path = "./duplicatesentences.json"
    print(f"layer={layer}, head={head}")

    x_ab, y_ab, x_ba, y_ba = scorer.collect_scatter_data(
    dataset_path,
    layer,
    head
    )

    scorer.plot_scatter(x_ab, y_ab, x_ba, y_ba, layer, head)