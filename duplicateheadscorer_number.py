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

        print("model loaded")
        print(f"layers: {self.num_layers}")
        print(f"heads: {self.num_heads}")

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

        points_ab = []
        points_ba = []

        used = 0
        skipped = 0

        for idx, item in enumerate(dataset):

            pair_id = item["id"]

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
                or len(repl_control) < 1
            ):

                skipped += 1

                print(f"\n[SKIP] Example {idx}")
                print("ID:", pair_id)
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

            points_ab.append({
                "x": probe_ab,
                "y": control_ab,
                "id": pair_id,
                "probe_sentence": probe,
                "control_sentence": control
            })

            probe_ba = att_p[j_p, i_p]
            control_ba = att_c[j_c, i_c]

            points_ba.append({
                "x": probe_ba,
                "y": control_ba,
                "id": pair_id,
                "probe_sentence": probe,
                "control_sentence": control
            })

            used += 1

        print(f"[Layer {layer} Head {head}]")
        print("used pairs:", used)
        print("skipped pairs:", skipped)

        return points_ab, points_ba

    def plot_scatter(self, points_ab, points_ba, layer, head):

        if len(points_ab) == 0:
            print(f"[Layer {layer} Head {head}] no data points")
            return

        fig, ax = plt.subplots(figsize=(10, 10))

        ax.scatter([], [], marker="o", color="C0", label="A → B")
        ax.scatter([], [], marker="+", color="C1", label="B → A")

        text_objects = []


        #this makes sure that the text is detectable by a mouse
        for point in points_ab:
            txt = ax.text(
                point["x"],
                point["y"],
                str(point["id"]),
                fontsize=9,
                color="C0", 
                ha="center",
                va="center",
                picker=True
            )
            text_objects.append((txt, point))

        for point in points_ba:
            txt = ax.text(
                point["x"],
                point["y"],
                str(point["id"]),
                fontsize=9,
                color="C1",  
                ha="center",
                va="center",
                picker=True
            )
            text_objects.append((txt, point))

        all_vals = []
        for p in points_ab:
            all_vals.extend([p["x"], p["y"]])
        for p in points_ba:
            all_vals.extend([p["x"], p["y"]])

        mn = min(all_vals)
        mx = max(all_vals)

        ax.plot(
            [mn, mx],
            [mn, mx],
            "--",
            color="black",
            alpha=0.5
        )

        ax.set_xlim(mn - (mx - mn) * 0.05, mx + (mx - mn) * 0.05)
        ax.set_ylim(mn - (mx - mn) * 0.05, mx + (mx - mn) * 0.05)

        ax.set_xlabel("Probe Attention")
        ax.set_ylabel("Control Attention")
        ax.set_title(f"Layer {layer}, Head {head}")

        ax.legend()
        ax.grid(True)
        #this makes an annotation that is invisible
        annot = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(
                boxstyle="round",
                fc="white",
                ec="black",
                alpha=0.95
            ),
            arrowprops=dict(
                arrowstyle="->"
            )
        )
        annot.set_visible(False)

        def update_annotation(point):
            annot.xy = (point["x"], point["y"])
            annot.set_text(
                f"ID: {point['id']}\n\n"
                f"Experimental:\n"
                f"{point['probe_sentence']}\n\n"
                f"Control:\n"
                f"{point['control_sentence']}"
            )

        def hover(event):
            if event.inaxes != ax:
                if annot.get_visible():
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
                return

            found = False
            #does the mouse hit the number, if so make the popup visible
            for text_obj, point in text_objects:
                contains, _ = text_obj.contains(event)
                if contains:
                    update_annotation(point)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    found = True
                    break

            if not found and annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()
        #this is a hover event listener
        fig.canvas.mpl_connect("motion_notify_event", hover)

        plt.tight_layout()

        os.makedirs("scatter_plots", exist_ok=True)
        output_path = f"scatter_plots/scatter_layer{layer}_head{head}.svg"

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()
        plt.close()

        print(f"saved plot to: {output_path}")

    def run_single_plot(
        self,
        dataset_path,
        layer=0,
        head=0
    ):

        print("\n" + "=" * 60)
        print(f"PROCESSING LAYER {layer} HEAD {head}")
        print("=" * 60)

        points_ab, points_ba = self.collect_scatter_data(
            dataset_path,
            layer,
            head
        )

        self.plot_scatter(
            points_ab,
            points_ba,
            layer,
            head
        )

        print("\nDONE")
        print("generated 1 scatter plot")


if __name__ == "__main__":

    scorer = DuplicateHeadScorer()
    dataset_path = "./duplicatesentences.json"

    layer = 2
    head = 8

    scorer.run_single_plot(
        dataset_path=dataset_path,
        layer=layer,
        head=head
    )