from pathlib import Path
import json
import matplotlib.pyplot as plt

from utils import ModelManager


class BatchAttentionExporter:
    def __init__(self, model_name="google/flan-t5-large"):
        self.model = ModelManager(model_name)
        self.model.load_model()

    def load_dataset(self, path):
        """
        expected JSON format:
        [
            {
                "id": "",
                "sentence1": "",
                "sentence2": "",
                "layer": _,
                "head": _
            },
            ...
        ]
        """
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def plot_and_save(self, attn1, attn2, diff, tokens1, tokens2, save_path):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(attn1)
        axs[0].set_title("Sentence 1 Attention")

        axs[1].imshow(attn2)
        axs[1].set_title("Sentence 2 Attention")

        axs[2].imshow(diff)
        axs[2].set_title("Difference")

        # FIX: use correct token sets per sentence
        axs[0].set_xticks(range(len(tokens1)))
        axs[0].set_yticks(range(len(tokens1)))
        axs[0].set_xticklabels(tokens1, rotation=90)
        axs[0].set_yticklabels(tokens1)

        axs[1].set_xticks(range(len(tokens2)))
        axs[1].set_yticks(range(len(tokens2)))
        axs[1].set_xticklabels(tokens2, rotation=90)
        axs[1].set_yticklabels(tokens2)

        # diff: use tokens1 (or you could build a combined alignment later)
        axs[2].set_xticks(range(len(tokens1)))
        axs[2].set_yticks(range(len(tokens1)))
        axs[2].set_xticklabels(tokens1, rotation=90)
        axs[2].set_yticklabels(tokens1)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    def run_batch(self, dataset_path, output_folder):
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        data = self.load_dataset(dataset_path)

        for item in data:
            print(f"Processing {item['id']}")

            outputs = self.model.get_attention_outputs(
                [item["sentence1"], item["sentence2"]]
            )

            attentions = outputs.encoder_attentions[item["layer"]]

            attn1 = attentions[0, item["head"]].detach().numpy()
            attn2 = attentions[1, item["head"]].detach().numpy()
            diff = attn1 - attn2

            inputs = self.model.tokenizer(
                [item["sentence1"], item["sentence2"]],
                padding=True,
                return_tensors="pt"
            )

            tokens1 = self.model.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
            tokens2 = self.model.tokenizer.convert_ids_to_tokens(inputs.input_ids[1])

            save_path = output_folder / f"{item['id']}_L{item['layer']}_H{item['head']}.png"

            self.plot_and_save(attn1, attn2, diff, tokens1, tokens2, save_path)


if __name__ == "__main__":
    exporter = BatchAttentionExporter()

    exporter.run_batch(
        dataset_path="",
        output_folder=""
    )